#!/usr/bin/env python3
"""
MiroFish Simulation Exporter
==============================
Reads a completed simulation directory and produces a single structured JSON
file suitable for analysis, visualization, or portfolio display.

Usage:
    python scripts/export_simulation.py <simulation_dir>
    python scripts/export_simulation.py <simulation_dir> -o report.json
    python scripts/export_simulation.py <simulation_dir> --pretty
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    if not os.path.exists(path):
        return records
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def load_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_actions(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze action records into summary statistics and timeline."""
    actions = [r for r in records if 'action_type' in r and 'event_type' not in r]
    events = [r for r in records if 'event_type' in r]

    if not actions:
        return {"total_actions": 0}

    action_types = Counter(a['action_type'] for a in actions)
    agents = defaultdict(lambda: {"actions": 0, "types": Counter()})
    rounds_data = defaultdict(lambda: {"actions": 0, "types": Counter()})
    posts = []

    for a in actions:
        agent_id = a.get('agent_id')
        agent_name = a.get('agent_name', f'agent_{agent_id}')
        action_type = a['action_type']
        round_num = a.get('round', 0)

        agents[agent_name]["actions"] += 1
        agents[agent_name]["types"][action_type] += 1
        agents[agent_name]["agent_id"] = agent_id

        rounds_data[round_num]["actions"] += 1
        rounds_data[round_num]["types"][action_type] += 1

        if action_type in ("CREATE_POST", "QUOTE_POST"):
            content = (a.get('action_args') or {}).get('content', '')
            if content:
                posts.append({
                    "round": round_num,
                    "agent_id": agent_id,
                    "agent_name": agent_name,
                    "type": action_type,
                    "content": content,
                })

    sim_start = next((e for e in events if e.get('event_type') == 'simulation_start'), None)
    sim_end = next((e for e in events if e.get('event_type') == 'simulation_end'), None)

    agent_summaries = []
    for name, data in sorted(agents.items(), key=lambda x: -x[1]["actions"]):
        agent_summaries.append({
            "agent_id": data["agent_id"],
            "name": name,
            "total_actions": data["actions"],
            "action_breakdown": dict(data["types"]),
        })

    timeline = []
    for round_num in sorted(rounds_data.keys()):
        rd = rounds_data[round_num]
        timeline.append({
            "round": round_num,
            "actions": rd["actions"],
            "types": dict(rd["types"]),
        })

    return {
        "total_actions": len(actions),
        "total_rounds": max(rounds_data.keys()) + 1 if rounds_data else 0,
        "action_types": dict(action_types),
        "agents": agent_summaries,
        "timeline": timeline,
        "posts": posts,
        "start_time": sim_start.get("timestamp") if sim_start else None,
        "end_time": sim_end.get("timestamp") if sim_end else None,
    }


def extract_agent_profiles(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract clean agent profile data from simulation config."""
    profiles = []
    for ac in config.get("agent_configs", []):
        profile = {
            "agent_id": ac.get("agent_id"),
            "name": ac.get("name", ""),
            "role": ac.get("role", ""),
            "archetype": ac.get("archetype", "contributor"),
            "interested_topics": ac.get("interested_topics", []),
        }
        if ac.get("persona"):
            profile["persona"] = ac["persona"]
        if ac.get("description"):
            profile["description"] = ac["description"]
        profiles.append(profile)
    return profiles


def export_simulation(sim_dir: str) -> Dict[str, Any]:
    """Build the full export from a simulation directory."""
    config = load_json(os.path.join(sim_dir, "simulation_config.json"))
    if not config:
        print(f"Error: No simulation_config.json found in {sim_dir}", file=sys.stderr)
        sys.exit(1)

    report: Dict[str, Any] = {
        "version": "0.5.0",
        "exported_at": datetime.now().isoformat(),
        "simulation_id": config.get("simulation_id", os.path.basename(sim_dir)),
    }

    # Simulation metadata
    report["metadata"] = {
        "requirement": config.get("simulation_requirement", ""),
        "project_id": config.get("project_id", ""),
        "graph_id": config.get("graph_id", ""),
        "time_config": config.get("time_config", {}),
    }

    # Agent profiles
    report["agents"] = extract_agent_profiles(config)

    # Platform results
    twitter_path = os.path.join(sim_dir, "twitter", "actions.jsonl")
    reddit_path = os.path.join(sim_dir, "reddit", "actions.jsonl")

    twitter_records = load_jsonl(twitter_path)
    reddit_records = load_jsonl(reddit_path)

    if twitter_records:
        report["twitter"] = analyze_actions(twitter_records)
    if reddit_records:
        report["reddit"] = analyze_actions(reddit_records)

    # Synthetic personas (if generated)
    personas = load_json(os.path.join(sim_dir, "synthetic_personas.json"))
    if personas:
        report["synthetic_personas"] = personas

    # Summary stats
    tw_total = report.get("twitter", {}).get("total_actions", 0)
    rd_total = report.get("reddit", {}).get("total_actions", 0)
    tw_posts = len(report.get("twitter", {}).get("posts", []))
    rd_posts = len(report.get("reddit", {}).get("posts", []))

    report["summary"] = {
        "total_actions": tw_total + rd_total,
        "total_posts": tw_posts + rd_posts,
        "platforms": [],
        "agent_count": len(report["agents"]),
    }
    if twitter_records:
        report["summary"]["platforms"].append("twitter")
    if reddit_records:
        report["summary"]["platforms"].append("reddit")

    return report


def main():
    parser = argparse.ArgumentParser(description='Export MiroFish simulation as structured JSON')
    parser.add_argument('simulation_dir', help='Path to simulation directory')
    parser.add_argument('-o', '--output', help='Output JSON path (default: <sim_dir>/export.json)')
    parser.add_argument('--pretty', action='store_true', help='Pretty-print JSON output')
    parser.add_argument('--no-posts', action='store_true', help='Exclude full post content (smaller file)')
    args = parser.parse_args()

    sim_dir = os.path.abspath(args.simulation_dir)
    if not os.path.isdir(sim_dir):
        print(f"Error: {sim_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    report = export_simulation(sim_dir)

    if args.no_posts:
        for platform in ("twitter", "reddit"):
            if platform in report:
                report[platform].pop("posts", None)

    indent = 2 if args.pretty else None
    report_json = json.dumps(report, indent=indent, ensure_ascii=False)

    output_path = args.output or os.path.join(sim_dir, "export.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_json)

    post_count = report["summary"]["total_posts"]
    action_count = report["summary"]["total_actions"]
    agent_count = report["summary"]["agent_count"]
    platforms = ", ".join(report["summary"]["platforms"])

    print(f"Exported: {action_count} actions, {post_count} posts, "
          f"{agent_count} agents ({platforms})")
    print(f"Saved to: {output_path}")


if __name__ == '__main__':
    main()
