"""Step 1 vs Step 2 (questions) and Step 3 (conversations) evaluation. No extra LLM calls: rule-based scoring
on must_include / must_not_include / tool calls / gold_evidence from the practice CSVs."""
import argparse, json, os, re, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from .config import DATASET, ROOT
from .agent import Agent, shared_kb
from .llm import LLM


def norm(s):
    s = str(s).lower().replace("’", "'")
    return re.sub(r"\s+", " ", re.sub(r"(?<=\d),(?=\d)", "", s)).strip()


def _spec(x):
    return [g.strip() for g in x.split(" ; ") if g.strip()] if isinstance(x, str) else []


def inc_ok(ans, spec):
    a = norm(ans)
    return all(any(norm(alt) in a for alt in g.split(" / ")) for g in _spec(spec))


def exc_ok(ans, spec):
    a = norm(ans)
    return not any(norm(alt) in a for g in _spec(spec) for alt in g.split(" / "))


STOP = threading.Event()
_FATAL = ("402", "401", "403", "credits", "quota", "per day", "perday", "invalid api key", "insufficient", "429")


def is_fatal(e):
    return any(x in str(e).lower() for x in _FATAL)


def run_cached(fn, items, key, cache_path, workers, fresh=False):
    """Run fn over items, appending each finished result to a jsonl cache so a rerun only does what is missing."""
    if fresh and cache_path.exists():
        cache_path.unlink()
    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            d = json.loads(line); cache[d["k"]] = d["v"]
    todo = [it for it in items if key(it) not in cache]
    STOP.clear()
    with ThreadPoolExecutor(workers) as ex, open(cache_path, "a", encoding="utf-8") as f:
        futs = {ex.submit(fn, it): it for it in todo}
        for fu in as_completed(futs):
            res = fu.result()
            if res is None:
                continue
            payload, ok = res
            if ok:
                cache[key(futs[fu])] = payload
                f.write(json.dumps({"k": key(futs[fu]), "v": payload}, ensure_ascii=False, default=str) + "\n"); f.flush()
            print(".", end="", flush=True)
    done = [cache[key(it)] for it in items if key(it) in cache]
    return done, len(items)


def kb_gold(x):
    return [g for g in _spec(x) if ":" not in g]


def _run_question(args):
    r, mode, use_asr, llm = args
    if STOP.is_set():
        return None
    ag = Agent(mode=mode, voice=use_asr, llm=llm)
    try:
        ans = ag.respond(r["asr_text"] if use_asr else r["question"])
    except Exception as e:
        if is_fatal(e):
            STOP.set(); print("\nSTOP:", str(e)[:420]); return None
        print("\nERROR:", str(e)[:420]); ans = f"[ERROR] {e}"
    gold, got = kb_gold(r["gold_evidence"]), set(ag.retrieved_ids)
    tix = bool(ag.trace and ag.trace[-1]["ticket_created"])
    row = {"id": r["id"], "lang": r["language"], "mode": mode, "expected": r["expected_decision"], "answer": ans,
           "include_ok": inc_ok(ans, r["must_include"]), "exclude_ok": exc_ok(ans, r["must_not_include"]),
           "ticket": tix, "retrieved": " ; ".join(ag.retrieved_ids),
           "recall": (sum(g in got for g in gold) / len(gold)) if gold and mode == "rag" else None}
    row["pass"] = row["include_ok"] and row["exclude_ok"] and (tix if r["expected_decision"] == "escalate" else True)
    return row, not ans.startswith("[ERROR]")


def eval_questions(mode, use_asr=False, limit=None, workers=4, fresh=False, ids=None):
    q = pd.read_csv(DATASET / "practice" / "questions.csv", dtype=str)
    q = q[q["id"].isin(ids)] if ids else q
    q = q.head(limit) if limit else q
    shared_kb(); llm = LLM()
    items = [(r, mode, use_asr, llm) for r in q.to_dict("records")]
    rows, total = run_cached(_run_question, items, lambda it: it[0]["id"], _cache(mode, use_asr, "_subset" if ids else ""), workers, fresh)
    return pd.DataFrame(rows), llm.usage, total


def _cache(what, asr=False, tag=""):
    d = ROOT / "eval_results"; d.mkdir(exist_ok=True)
    return d / f"cache_{what}{'_asr' if asr else ''}{tag}.jsonl"


def _parse_calls(s):
    out = []
    for m in re.finditer(r"(\w+)\(([^)]*)\)", s) if isinstance(s, str) else []:
        out.append((m.group(1), dict(re.findall(r"(\w+)=([^,\s)]+)", m.group(2)))))
    return out


def _call_matches(exp, calls):
    name, kv = exp
    for c in calls:
        if c["name"] != name:
            continue
        if name == "lookup_order":
            oid = "".join(ch for ch in str(c["args"].get("order_id", "")) if ch.isdigit())
            vv = str(c["args"].get("verification_value") or "")
            if oid == kv.get("order_id", oid) and ("verification_value" in kv) == bool(vv) and (not vv or vv == kv["verification_value"]):
                return True
        else:
            return True
    return False


def _run_conv(args):
    cid, turns, llm = args
    if STOP.is_set():
        return None
    ag, rows, clean = Agent(mode="rag", llm=llm), [], True
    for t in turns:
        try:
            ans = ag.respond(t["customer_message"])
        except Exception as e:
            if is_fatal(e):
                STOP.set(); print("\nSTOP:", str(e)[:420]); return None
            print("\nERROR:", str(e)[:420]); ans = f"[ERROR] {e}"; clean = False
        tr = ag.trace[-1] if ag.trace else {"tool_calls": [], "ticket_created": False}
        exp_calls = _parse_calls(t["expected_tool_calls"])
        calls_ok = all(_call_matches(e, tr["tool_calls"]) for e in exp_calls)
        tix_ok = tr["ticket_created"] == any(n == "create_ticket" for n, _ in exp_calls)
        rows.append({"conv": cid, "turn": t["turn"], "scenario": t["scenario"], "expected": t["expected_decision"],
                     "user": t["customer_message"], "answer": ans, "include_ok": inc_ok(ans, t["must_include"]),
                     "exclude_ok": exc_ok(ans, t["must_not_include"]), "tools_ok": calls_ok, "ticket_ok": tix_ok,
                     "retrieved": " ; ".join(ag.retrieved_ids)})
        rows[-1]["pass"] = all(rows[-1][k] for k in ("include_ok", "exclude_ok", "tools_ok", "ticket_ok"))
    return rows, clean


def eval_conversations(limit=None, workers=4, fresh=False, ids=None):
    c = pd.read_csv(DATASET / "practice" / "conversations.csv", dtype=str)
    c["turn"] = c["turn"].astype(int)
    groups = [(cid, g.sort_values("turn").to_dict("records")) for cid, g in c.groupby("conversation_id")]
    groups = [g for g in groups if g[0] in ids] if ids else groups
    groups = groups[:limit] if limit else groups
    shared_kb(); llm = LLM()
    parts, total = run_cached(_run_conv, [(i, t, llm) for i, t in groups], lambda it: it[0], _cache("convs", False, "_subset" if ids else ""), workers, fresh)
    return pd.DataFrame([r for part in parts for r in part]), llm.usage, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["baseline", "rag", "convs", "all"])
    ap.add_argument("--asr", action="store_true", help="use asr_text instead of question (simulates STT)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--ids", help="comma-separated question/conversation ids to rerun only (e.g. PQ0008,PQ0022 or PC002,PC010); saves tokens")
    ap.add_argument("--fresh", action="store_true", help="ignore cached results and rerun everything (use after changing code/prompts)")
    ap.add_argument("--workers", type=int, default=int(os.getenv("EVAL_WORKERS", 4)))
    a = ap.parse_args()
    out = ROOT / "eval_results"; out.mkdir(exist_ok=True)
    ids = [x.strip() for x in a.ids.split(",") if x.strip()] if a.ids else None
    suffix = "_subset" if ids else ""
    sj = out / "summary.json"
    summary = json.loads(sj.read_text()) if (sj.exists() and not ids) else {}
    for what in (["baseline", "rag", "convs"] if a.what == "all" else [a.what]):
        if what == "convs":
            df, use, total = eval_conversations(a.limit, a.workers, a.fresh, ids)
            if df.empty:
                print(f"\n{what}: nothing completed (0/{total}); fix the error above and rerun."); continue
            summary[what] = {"turn_pass": round(df["pass"].mean(), 3), "conv_pass": round(df.groupby("conv")["pass"].all().mean(), 3),
                             "include": round(df.include_ok.mean(), 3), "tools": round(df.tools_ok.mean(), 3), "ticket": round(df.ticket_ok.mean(), 3), "usage": use}
        else:
            df, use, total = eval_questions(what, a.asr, a.limit, a.workers, a.fresh, ids)
            if df.empty:
                print(f"\n{what}: nothing completed (0/{total}); fix the error above and rerun."); continue
            summary[what] = {"pass": round(df["pass"].mean(), 3), "include": round(df.include_ok.mean(), 3),
                             "avoid_banned": round(df.exclude_ok.mean(), 3), "usage": use}
            if what == "rag":
                summary[what]["retrieval_recall"] = round(df.recall.dropna().mean(), 3)
        n = df["conv"].nunique() if what == "convs" else len(df)
        summary[what]["completed"] = f"{n}/{total}"
        if n < total:
            print(f"\n{what}: PARTIAL {n}/{total} done. Rerun the same command to continue (results are saved).")
        df.to_csv(out / f"{what}{'_asr' if a.asr else ''}{suffix}.csv", index=False)
    if not ids:
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
