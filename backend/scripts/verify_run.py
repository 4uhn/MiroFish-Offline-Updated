"""Verify a finished run: simulation grounding, graph memory, Ollama context use
and the report's quotes, numbers and claims, in one pass.

Usage (from backend/):
    .venv/bin/python scripts/verify_run.py <backend_log> <ollama_log>
        [--sim sim_...] [--report report_...] [--ollama-offset BYTES]

IDs default to the last ones named in the backend log. --ollama-offset skips
the start of a shared Ollama log (its size in bytes when the run began).
Never prints secrets: Neo4j credentials come from app.config.Config.
"""
import argparse, ast, json, re, sqlite3, sys
from collections import Counter
from datetime import datetime

sys.path.insert(0, '.')
ap = argparse.ArgumentParser()
ap.add_argument("backend_log"); ap.add_argument("ollama_log")
ap.add_argument("--sim"); ap.add_argument("--report"); ap.add_argument("--ollama-offset", type=int, default=0)
args = ap.parse_args()

blog = open(args.backend_log).read()
blines = blog.splitlines()
last = lambda pat: (re.findall(pat, blog) or [None])[-1]
SIM = args.sim or last(r'Created simulation: (sim_[0-9a-f]+)')
REP = args.report or last(r'Report generation complete: (report_[0-9a-f]+)')
# a regeneration's log has no "Created project" line; the simulation records it
PROJ = json.load(open(f"uploads/simulations/{SIM}/state.json")).get("project_id") \
    or last(r'Created project: (proj_[0-9a-f]+)')
print(f"project={PROJ} sim={SIM} report={REP}\n")

def hdr(t): print(f"\n=== {t} ===")
norm = lambda t: re.sub(r"\s+", " ", (t or "").replace("\u2019", "'").replace("\u2018", "'")
                        .replace("\u201c", '"').replace("\u201d", '"')).strip().lower()
stamp = lambda l: datetime.strptime(l[1:9], "%H:%M:%S") if re.match(r"\[\d\d:\d\d:\d\d\]", l) else None

# The report's own slice of the backend log: a regenerated report shares the
# log with the original run, so its phase starts at the last "Measured
# requirement" line before its "Report generation complete" line.
rep_end = next((i for i in range(len(blines) - 1, -1, -1)
                if REP and f"Report generation complete: {REP}" in blines[i]), None)
rep_start = None
if rep_end is not None:
    rep_start = next((i for i in range(rep_end, -1, -1)
                      if re.search(r"Measured requirement keywords|Starting report generation", blines[i])), None)
rep_log = blines[rep_start:rep_end + 1] if rep_start is not None else []

# ---------------------------------------------------------------- phases
hdr("Phase timings (backend log)")
def ts(pat, first=True):
    hits = [l for l in blines if re.search(pat, l) and stamp(l)]
    return stamp(hits[0] if first else hits[-1]) if hits else None
marks = [
    ("ontology", r"=== Starting ontology generation", r"Ontology complete"),
    ("graph", r"=== Starting graph build", r"Graph build complete"),
    ("profiles+config", r"Created simulation:", r"Simulation config generation complete"),
    ("simulation", r"Simulation config generation complete", r"All platform simulations completed"),
    ("drain", r"draining graph memory", r"Graph memory drained"),
]
t0 = ts(marks[0][1])
for name, a, b in marks:
    s, e = ts(a), (ts(b) if name == "drain" else ts(b, False))
    print(f"{name:16} {str(e - s) if s and e else '?'}")
rs = next((stamp(l) for l in rep_log if stamp(l)), None)
re_ = next((stamp(l) for l in reversed(rep_log) if stamp(l)), None)
print(f"{'report':16} {str(re_ - rs) if rs and re_ else '?'}")
if t0 and re_ and t0 < re_:
    print(f"{'TOTAL':16} {re_ - t0}  (includes any gap before a regenerated report)")

# ---------------------------------------------------------------- simulation
simdir = f"uploads/simulations/{SIM}"
slog = open(f"{simdir}/simulation.log").read()
hdr("Simulation summary (System One, pool)")
for l in re.findall(r".*done: .*System One.*", slog): print(l[:260])
tot = [tuple(map(int, m)) for m in re.findall(r"System One: (\d+)/(\d+)", slog)]
if tot: print(f"S1 overall: {sum(a for a,_ in tot)}/{sum(b for _,b in tot)} = {100*sum(a for a,_ in tot)/sum(b for _,b in tot):.0f}%")

hdr("Scheduled events fired")
fired = re.findall(r"\[(\w+)\] Scheduled event fired for agent \d+: (.*)", slog)
for p, d in sorted(set(fired)): print(f"{p:8} {d[:90]}")
cfg = json.load(open(f"{simdir}/simulation_config.json"))
print(f"configured events: {len(cfg.get('event_config', cfg).get('scheduled_events', cfg.get('scheduled_events', [])))}")

# ---------------------------------------------------------------- texts
rows = []   # (platform, kind, author display name, text)
display = {int(p["user_id"]): p.get("name") or p.get("username")
           for p in json.load(open(f"{simdir}/reddit_profiles.json"))}
for plat in ("twitter", "reddit"):
    db = sqlite3.connect(f"{simdir}/{plat}_simulation.db")
    users = {u: display.get(u, n) for u, n in db.execute("select user_id, name from user")}
    q = [("post", "select user_id, content from post where content!=''"),
         ("quote", "select user_id, quote_content from post where quote_content is not null"),
         ("comment", "select user_id, content from comment")]
    for kind, sql in q:
        rows += [(plat, kind, users.get(u), c) for u, c in db.execute(sql)]
print(f"\nagent texts: {len(rows)} ({dict(Counter(k for _,k,_,_ in rows))})")

seed = open(f"uploads/projects/{PROJ}/extracted_text.txt").read()
num_re = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?%?")
canon = lambda n: n.replace(",", "").rstrip("%")
seed_nums = {canon(n) for n in num_re.findall(seed)}
hdr("Numbers in agent texts not in seed")
bad = Counter()
for _, _, a, t in rows:
    for n in num_re.findall(t or ""):
        if canon(n) not in seed_nums: bad[(canon(n), a)] += 1
print(dict(bad) or "none")

# ---------------------------------------------------------------- graph memory
hdr("Graph memory grounding")
drops = re.findall(r"NER grounding: dropped (\d+) unsupported relations", blog)
edrops = re.findall(r"NER grounding: dropped (\d+) entities", blog)
print(f"relation-drop log lines: {len(drops)}, relations dropped: {sum(map(int, drops))}; entity drops: {sum(map(int, edrops))}")
try:
    from app.config import Config
    from neo4j import GraphDatabase
    drv = GraphDatabase.driver(Config.NEO4J_URI, auth=(Config.NEO4J_USER, Config.NEO4J_PASSWORD))
    with drv.session() as s:
        r = s.run("MATCH ()-[r:RELATION]->() WHERE r.simulation_id=$sim RETURN r.name AS n, count(*) AS c ORDER BY c DESC", sim=SIM)
        rel = [(x["n"], x["c"]) for x in r]
        print(f"sim edges in Neo4j: {sum(c for _, c in rel)}  by type: {rel}")
    drv.close()
except Exception as e:
    print("neo4j check skipped:", type(e).__name__)

# ---------------------------------------------------------------- ollama
hdr("Ollama prompts")
with open(args.ollama_log, errors="ignore") as f:
    f.seek(args.ollama_offset)
    olog = f.read()
ntok = sorted(int(x) for x in re.findall(r"new prompt, n_ctx_slot = \d+, n_keep = \d+, task.n_tokens = (\d+)", olog))
if ntok:
    pct = lambda p: ntok[min(len(ntok) - 1, int(len(ntok) * p))]
    print(f"prompts={len(ntok)} p50={pct(.5)} p90={pct(.9)} max={ntok[-1]}")
lines = olog.splitlines()
for i, l in enumerate(lines):
    if "truncated = 1" in l:
        gin = next((x for x in lines[i:i + 60] if "[GIN]" in x), "")
        print("TRUNCATED:", l.split("|", 1)[-1].strip()[:90], "|", gin[6:40])
print("truncated=1 count:", olog.count("truncated = 1"))

if not REP:
    sys.exit("no report yet")

# ---------------------------------------------------------------- report log
hdr("Report pipeline (backend log, this report only)")
grab = lambda pat: [l for l in rep_log if re.search(pat, l)]
print(f"'probably dropped' warnings: {len(grab('probably dropped'))}")
trims = grab(r"trimmed \d+ chars")
print(f"context trims: {len(trims)}")
toks = [int(x) for l in grab(r"prompt \d+ tokens") for x in re.findall(r"prompt (\d+) tokens", l)]
if toks: print(f"section prompts: {len(toks)}, max {max(toks)} tokens")
for l in grab(r"quotes kept="):
    m = re.search(r"Section (.{,40}?): quotes kept=(\d+), dropped=(\d+).*misattributed=(\d+), inline_dropped=(\d+)", l)
    if m: print(f"  quotes | {m[1]:40} kept={m[2]} dropped={m[3]} misattributed={m[4]} inline_dropped={m[5]}")
summ = grab(r"Report summary:")
print("post-hoc summary:", (summ[-1].split("Report summary:", 1)[1].strip()[:160] if summ else "MISSING"))

# ---------------------------------------------------------------- report text
repdir = f"uploads/reports/{REP}"
rep = open(f"{repdir}/full_report.md").read()
alog = [json.loads(l) for l in open(f"{repdir}/agent_log.jsonl")]
hdr("Report tool use")
calls = Counter(r["details"].get("tool_name") for r in alog if r["action"] == "tool_call")
print(dict(calls))
print("titles:", [l[3:] for l in rep.splitlines() if l.startswith("## ")])

# interview answers per agent, from the tool results the sections saw
interviews = {}   # name -> normalised answers
for r in alog:
    if r["action"] == "tool_result" and r["details"].get("tool_name") == "interview_agents":
        for block in re.split(r"\n#### Interview #\d+: ", r["details"].get("result", ""))[1:]:
            name, _, body = block.partition("\n")
            interviews[name.strip()] = interviews.get(name.strip(), "") + " " + norm(body)
print(f"interviewed agents with answers: {len(interviews)}")

from app.services.report_agent import ReportAgent as RA
corpus = [norm(t) for *_, t in rows]
names = {v for v in display.values() if v} | set(interviews)
parts_of = lambda q: [norm(p) for p in re.split(r"\.\.\.|\u2026", q) if p.strip()]

def source(q):
    """(where the quote is word for word, the agents who wrote or said it)"""
    ps = parts_of(q)
    db = {a for _, _, a, t in rows if all(p in norm(t) for p in ps)}
    if db: return "DB", db
    iv = {n for n, t in interviews.items() if all(p in t for p in ps)}
    return ("INTERVIEW", iv) if iv else ("MISSING", set())

hdr("Quotes: verbatim, attribution, interview framing, repeats")
per_speaker, seen, problems = Counter(), Counter(), 0
prev = ""   # the prose line before a block quote, which often credits it ("Priya Davies wrote:")
for line in rep.splitlines():
    block = line.startswith(">")
    if line.strip() and not block:
        prev = line
    for m in RA._INLINE_QUOTE.finditer(line):
        q = m.group(1)
        if len(q.split()) < 4: continue
        src, authors = source(q)
        context = (line[m.end():m.end() + 120] + " " + prev[-200:]) if block else line[max(0, m.start() - 200):m.start()]
        credited = {n for n in names if n in context}
        attr_ok = not credited or bool(credited & authors)
        framed = src != "INTERVIEW" or ("interview" in context.lower() if block else
                                        re.search(r"(?i)interview|told", line) is not None)
        seen[norm(q)] += 1
        if block: per_speaker[next(iter(credited & authors), "uncredited")] += 1
        ok = src != "MISSING" and attr_ok and framed
        problems += not ok
        flag = "" if ok else f"  <-- {'not verbatim' if src == 'MISSING' else ''}{'' if attr_ok else f' credited={sorted(credited)} actual={sorted(authors)}'}{'' if framed else ' interview answer not framed as one'}"
        print(f"{'OK ' if ok else 'BAD'} {'block ' if block else 'inline'} {src:9} {q[:60]!r}{flag}")
rep_quotes = [k for k, c in seen.items() if c > 1]
print(f"problems: {problems} | block quotes per speaker: {dict(per_speaker)} | repeated quotes: {len(rep_quotes)}")
print("interview credits '(…, interview)':", len(re.findall(r"\([^)]*\binterview\)", rep)),
      "| 'told interviewers':", len(re.findall(r"(?i)told interviewers", rep)))

hdr("Numbers in report not in seed or measured counts")
kw = next((ast.literal_eval(m[1]) for l in reversed(rep_log) for m in [re.search(r"Measured requirement keywords: (\[.*\])", l)] if m), None)
meas = ""
if kw:
    from app.services.agent_posts import AgentPostIndex
    meas = AgentPostIndex.for_simulation(SIM).stats_text(kw)
    print("measured counts:\n" + meas)
stats_txt = " ".join(json.dumps(r["details"].get("result", "")) for r in alog if r["action"] == "tool_result")
allowed = seed_nums | {canon(n) for n in num_re.findall(meas + stats_txt)}
rbad = Counter(canon(n) for n in num_re.findall(rep) if canon(n) not in allowed)
print(dict(rbad) or "none")

hdr("Blame / relation claims in report (review by hand)")
for s in re.split(r"(?<=[.!?])\s+|\n+", rep):
    if re.search(r"(?i)\bblam|\baccus|\bcriticis|\bholds? .* responsible", s): print("-", s.strip()[:220])
