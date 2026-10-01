# Security

## Scope

MiroFish Offline+ is a single-user research tool that runs entirely on your own machine. The backend (port 5001), the frontend (port 3000) and Neo4j bind to `127.0.0.1` only, and the API has no authentication. Don't expose these ports to a network.

## Reporting a vulnerability

Please report vulnerabilities privately through **Report a vulnerability** on this repository's Security tab, rather than in a public issue.

## Known dependency advisories

`transformers` is held at 4.x on purpose. OASIS's Twitter recommender uses [`Twitter/twhin-bert-base`](https://huggingface.co/Twitter/twhin-bert-base), whose relative position embeddings transformers 5 no longer supports. In transformers 5 the model still loads, but its embeddings come out different.

The open advisories against transformers 4.x all need you to load, save, convert or train on an untrusted model. This project only loads `Twitter/twhin-bert-base` and runs it forward:

| Advisory | Affected component | Used here? |
|---|---|---|
| CVE-2025-14929 | X-CLIP checkpoint conversion | No |
| CVE-2026-1839 | `Trainer` checkpoint RNG restore | No |
| CVE-2026-5241 | LightGlue model loading | No |
| CVE-2026-9856 | `save_pretrained()` | No |
| CVE-2026-80047 | `load_custom_generate()` | No |
| CVE-2026-4372 | Loading a malicious `config.json` | Only `Twitter/twhin-bert-base` is loaded |
