#!/usr/bin/env python3
"""
MiroFish Benchmark Suite
=========================
Measures LLM throughput, embedding speed, model routing, and system resource
usage. Outputs a structured JSON report for analysis and portfolio display.

Usage:
    python scripts/benchmark.py                  # full suite
    python scripts/benchmark.py --skip-embedding # skip embedding bench
    python scripts/benchmark.py --output bench.json  # custom output path
"""

import argparse
import json
import os
import platform
import re
import resource
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), '../../.env')
if os.path.exists(env_path):
    load_dotenv(env_path, override=True)

from app.utils.llm_client import LLMClient
from app.utils.model_router import get_router, TaskType

_GENERATION_PROMPT = (
    "You are a Twitter user who cares about climate change. "
    "Write a short tweet (under 280 characters) about renewable energy progress."
)

_EXTRACTION_PROMPT = {
    "messages": [
        {"role": "system", "content": (
            "Extract entities from the text as JSON: "
            '{"entities": [{"name": "...", "type": "..."}]}'
        )},
        {"role": "user", "content": (
            "Tesla announced a new solar panel factory in Austin, Texas. "
            "CEO Elon Musk said production would begin in Q2 2026."
        )},
    ]
}


def _get_system_info() -> Dict[str, Any]:
    info = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
    }
    try:
        import psutil
        mem = psutil.virtual_memory()
        info["ram_gb"] = round(mem.total / (1024**3), 1)
    except ImportError:
        pass
    return info


def _get_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_maxrss / (1024 * 1024) if sys.platform == 'darwin' else usage.ru_maxrss / 1024


def _raw_completion(client: LLMClient, messages, temperature=0.7, max_tokens=256):
    """Call the OpenAI client directly to get usage stats alongside the response."""
    kwargs = {
        "model": client.model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if client._is_ollama():
        extra = {"options": {"num_ctx": client._num_ctx}, "think": False}
        kwargs["extra_body"] = extra
    return client.client.chat.completions.create(**kwargs)


def bench_llm_generation(client: LLMClient, model_name: str, n_runs: int = 3) -> Dict[str, Any]:
    """Benchmark text generation throughput."""
    timings = []
    token_counts = []
    prompt_token_counts = []
    last_response = ""

    for i in range(n_runs):
        messages = [
            {"role": "system", "content": "You are a social media user. Be concise."},
            {"role": "user", "content": _GENERATION_PROMPT},
        ]
        t0 = time.perf_counter()
        resp = _raw_completion(client, messages, temperature=0.7, max_tokens=256)
        elapsed = time.perf_counter() - t0
        timings.append(elapsed)

        content = resp.choices[0].message.content or ""
        content = re.sub(r'<think>[\s\S]*?</think>', '', content).strip()

        usage = resp.usage
        if usage and usage.completion_tokens:
            token_counts.append(usage.completion_tokens)
            prompt_token_counts.append(usage.prompt_tokens or 0)
        else:
            token_counts.append(len(content.split()))
            prompt_token_counts.append(0)
        last_response = content

    avg_time = sum(timings) / len(timings)
    avg_tokens = sum(token_counts) / len(token_counts)
    tok_per_sec = avg_tokens / avg_time if avg_time > 0 else 0

    return {
        "model": model_name,
        "task": "generation",
        "runs": n_runs,
        "avg_time_s": round(avg_time, 2),
        "avg_completion_tokens": round(avg_tokens, 1),
        "avg_prompt_tokens": round(sum(prompt_token_counts) / len(prompt_token_counts), 1),
        "tok_per_sec": round(tok_per_sec, 1),
        "sample_output": last_response[:200],
    }


def bench_llm_extraction(client: LLMClient, model_name: str, n_runs: int = 3) -> Dict[str, Any]:
    """Benchmark structured extraction (JSON mode)."""
    timings = []
    token_counts = []
    successes = 0

    for i in range(n_runs):
        messages = list(_EXTRACTION_PROMPT["messages"])
        t0 = time.perf_counter()
        try:
            resp = _raw_completion(client, messages, temperature=0.1, max_tokens=512)
            elapsed = time.perf_counter() - t0

            content = resp.choices[0].message.content or ""
            content = re.sub(r'<think>[\s\S]*?</think>', '', content).strip()
            content = re.sub(r'^```(?:json)?\s*\n?', '', content, flags=re.IGNORECASE)
            content = re.sub(r'\n?```\s*$', '', content).strip()
            result = json.loads(content)
            if isinstance(result, dict) and "entities" in result:
                successes += 1

            usage = resp.usage
            token_counts.append(usage.completion_tokens if usage else 0)
        except Exception:
            elapsed = time.perf_counter() - t0
            token_counts.append(0)
        timings.append(elapsed)

    avg_time = sum(timings) / len(timings)
    avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
    tok_per_sec = avg_tokens / avg_time if avg_time > 0 else 0

    return {
        "model": model_name,
        "task": "extraction",
        "runs": n_runs,
        "avg_time_s": round(avg_time, 2),
        "avg_completion_tokens": round(avg_tokens, 1),
        "tok_per_sec": round(tok_per_sec, 1),
        "success_rate": round(successes / n_runs, 2),
    }


def bench_embedding(n_texts: int = 20) -> Optional[Dict[str, Any]]:
    """Benchmark embedding throughput."""
    try:
        from app.storage.embedding_service import EmbeddingService
        svc = EmbeddingService()
        if not svc.health_check():
            return {"status": "unavailable", "reason": "health check failed"}
    except Exception as e:
        return {"status": "unavailable", "reason": str(e)}

    texts = [
        f"This is test sentence number {i} about topic {['climate', 'health', 'tech', 'politics'][i % 4]}"
        for i in range(n_texts)
    ]

    # Single embeddings
    t0 = time.perf_counter()
    for text in texts:
        svc.embed(text)
    single_elapsed = time.perf_counter() - t0

    # Batch embedding
    svc._cache.clear()
    t0 = time.perf_counter()
    svc.embed_batch(texts)
    batch_elapsed = time.perf_counter() - t0

    return {
        "status": "ok",
        "model": svc.model,
        "n_texts": n_texts,
        "single_total_s": round(single_elapsed, 2),
        "single_per_text_ms": round(single_elapsed / n_texts * 1000, 1),
        "batch_total_s": round(batch_elapsed, 2),
        "batch_per_text_ms": round(batch_elapsed / n_texts * 1000, 1),
        "batch_speedup": round(single_elapsed / batch_elapsed, 1) if batch_elapsed > 0 else 0,
    }


def bench_routing() -> Dict[str, Any]:
    """Report model routing configuration and tier availability."""
    router = get_router()
    return {
        "fast_model": router.fast_model_name if router.fast_available else None,
        "quality_model": router.quality_model_name,
        "fast_available": router.fast_available,
        "tier_map": {
            "extraction": "fast" if router.fast_available else "quality",
            "generation": "quality",
            "analysis": "quality",
        },
    }


def bench_response_pool(with_embedder: bool = True) -> Dict[str, Any]:
    """Benchmark response pool add/reuse cycle."""
    from app.services.response_pool import ResponsePool

    embedder = None
    if with_embedder:
        try:
            from app.storage.embedding_service import EmbeddingService
            embedder = EmbeddingService()
            if not embedder.health_check():
                embedder = None
        except Exception:
            pass

    pool = ResponsePool(reuse_probability=1.0, embedding_service=embedder)

    sample_posts = [
        ("contributor", "Renewable energy is making incredible progress. Solar costs dropped 90% in a decade."),
        ("amplifier", "Breaking: New study shows wind power now cheapest energy source in most regions!"),
        ("debater", "While solar adoption grows, we need to address battery storage challenges head on."),
        ("contributor", "Community solar programs are expanding access to clean energy for renters and low-income families."),
        ("lurker", "Interesting developments in green hydrogen — could be the missing piece for heavy industry."),
    ]

    t0 = time.perf_counter()
    for arch, text in sample_posts:
        pool.add(arch, text)
    add_time = time.perf_counter() - t0

    reuse_attempts = 20
    hits = 0
    t0 = time.perf_counter()
    for i in range(reuse_attempts):
        result = pool.try_reuse("contributor", query="solar energy renewable")
        if result:
            hits += 1
    reuse_time = time.perf_counter() - t0

    return {
        "has_embedder": embedder is not None,
        "add_time_ms": round(add_time * 1000, 1),
        "entries": pool.stats["pooled"],
        "deduped": pool.stats["deduped"],
        "reuse_attempts": reuse_attempts,
        "reuse_hits": hits,
        "semantic_matches": pool.stats["semantic_matches"],
        "reuse_time_ms": round(reuse_time * 1000, 1),
    }


def run_benchmark(skip_embedding: bool = False) -> Dict[str, Any]:
    """Run the full benchmark suite."""
    print("MiroFish Benchmark Suite")
    print("=" * 40)

    report: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "system": _get_system_info(),
    }

    # Model routing info
    print("\n[1/5] Model routing config...")
    report["routing"] = bench_routing()
    router = get_router()
    print(f"  Quality: {router.quality_model_name}")
    if router.fast_available:
        print(f"  Fast:    {router.fast_model_name}")
    else:
        print("  Fast:    not configured")

    # LLM generation benchmark (quality model)
    print("\n[2/5] LLM generation benchmark (quality model)...")
    quality_client = router.get_quality_client()
    gen_result = bench_llm_generation(quality_client, router.quality_model_name)
    report["generation"] = gen_result
    print(f"  {gen_result['tok_per_sec']} tok/s avg over {gen_result['runs']} runs")

    # LLM extraction benchmark (fast model if available, else quality)
    print("\n[3/5] LLM extraction benchmark...")
    ext_client = router.get_client(TaskType.EXTRACTION)
    ext_model = router.fast_model_name if router.fast_available else router.quality_model_name
    ext_result = bench_llm_extraction(ext_client, ext_model)
    report["extraction"] = ext_result
    print(f"  {ext_result['avg_time_s']}s avg, {ext_result['success_rate']*100:.0f}% success")

    # Fast model generation (if available, for comparison)
    if router.fast_available:
        print("\n[3b] Fast model generation (comparison)...")
        fast_client = router.get_client(TaskType.EXTRACTION)
        fast_gen = bench_llm_generation(fast_client, router.fast_model_name)
        report["generation_fast"] = fast_gen
        print(f"  {fast_gen['tok_per_sec']} tok/s avg")

    # Embedding benchmark
    if not skip_embedding:
        print("\n[4/5] Embedding benchmark...")
        emb_result = bench_embedding()
        report["embedding"] = emb_result
        if emb_result.get("status") == "ok":
            print(f"  Single: {emb_result['single_per_text_ms']}ms/text, "
                  f"Batch: {emb_result['batch_per_text_ms']}ms/text "
                  f"({emb_result['batch_speedup']}x speedup)")
        else:
            print(f"  Skipped: {emb_result.get('reason', 'unavailable')}")
    else:
        print("\n[4/5] Embedding benchmark... skipped")
        report["embedding"] = {"status": "skipped"}

    # Response pool benchmark
    print("\n[5/5] Response pool benchmark...")
    pool_result = bench_response_pool(with_embedder=not skip_embedding)
    report["response_pool"] = pool_result
    print(f"  {pool_result['entries']} entries, "
          f"{pool_result['reuse_hits']}/{pool_result['reuse_attempts']} reuse hits, "
          f"{pool_result['semantic_matches']} semantic")

    # Memory usage
    report["peak_rss_mb"] = round(_get_rss_mb(), 1)
    print(f"\nPeak RSS: {report['peak_rss_mb']} MB")

    return report


def main():
    parser = argparse.ArgumentParser(description='MiroFish Benchmark Suite')
    parser.add_argument('--skip-embedding', action='store_true', help='Skip embedding benchmarks')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output JSON path (default: prints to stdout)')
    parser.add_argument('--runs', type=int, default=3, help='Number of LLM benchmark runs per test')
    args = parser.parse_args()

    report = run_benchmark(skip_embedding=args.skip_embedding)

    report_json = json.dumps(report, indent=2, ensure_ascii=False)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(report_json)
        print(f"\nReport saved to {args.output}")
    else:
        print("\n" + "=" * 40)
        print("BENCHMARK REPORT")
        print("=" * 40)
        print(report_json)


if __name__ == '__main__':
    main()
