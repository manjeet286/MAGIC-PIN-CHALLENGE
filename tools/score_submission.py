"""Score submission.jsonl with the local judge's rubric (patched to see full context), through the bot's
cached, cost-logged LLM client. Re-scoring an unchanged message is free.

Usage: python tools/score_submission.py [--pairs T01,T21]
Writes per-pair reasons and hints to .cache/scores.md.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

DIMS = [("specificity", "Specificity"), ("category_fit", "Category fit"), ("merchant_fit", "Merchant fit"),
        ("decision_quality", "Trigger/decision"), ("engagement_compulsion", "Engagement")]


def make_scorer():
    """The local judge's LLMScorer, routed through bot.llm_json (cached + cost-logged)."""
    import bot
    import judge_simulator as js

    js.print_llm = lambda *a, **k: None

    class CachedProvider(js.LLMProvider):
        def name(self) -> str:
            return f"cached {bot.LLM_MODEL}"

        def complete(self, prompt: str, system: str = None) -> str:
            resp = bot.llm_json([{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                                "judge", max_tokens=900)
            if resp is None:
                raise RuntimeError("judge LLM unavailable")
            return json.dumps(resp)

    return js.LLMScorer(CachedProvider(), None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs")
    args = ap.parse_args()

    import bot
    from load_dataset import load_dataset

    data = load_dataset("expanded")
    cats, merchants = dict(data["category"]), dict(data["merchant"])
    customers, triggers = dict(data["customer"]), dict(data["trigger"])
    pairs = {p["test_id"]: p for p in json.loads((ROOT / "dataset/expanded/test_pairs.json").read_text(encoding="utf-8"))["pairs"]}
    subs = [json.loads(l) for l in (ROOT / "submission.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.pairs:
        wanted = set(args.pairs.split(","))
        subs = [s for s in subs if s["test_id"] in wanted]

    scorer = make_scorer()
    spent0 = bot.llm_spent()
    results, report = [], ["# Local judge scores\n"]
    for s in subs:
        p = pairs[s["test_id"]]
        m = merchants[p["merchant_id"]]
        trg = triggers[p["trigger_id"]]
        sc = scorer.score(s, cats[m["category_slug"]], m, trg, customers.get(p["customer_id"]))
        failed = "LLM scoring failed" in sc.hint
        results.append((s["test_id"], trg["kind"], sc, failed))
        dims = " ".join(f"{getattr(sc, k):>2}" for k, _ in DIMS)
        print(f"{s['test_id']} {trg['kind'][:24]:24} {dims}  = {sc.total:2}/50{'  (FALLBACK, not LLM-scored)' if failed else ''}")
        report.append(f"## {s['test_id']} {trg['kind']}: {sc.total}/50\n\n{s['body']}\n")
        for k, label in DIMS:
            reason = getattr(sc, "engagement_reason" if k == "engagement_compulsion" else f"{k}_reason")
            report.append(f"- **{label} {getattr(sc, k)}**: {reason}")
        report.append(f"- hint: {sc.hint}\n")

    scored = [r for r in results if not r[3]]
    if scored:
        n = len(scored)
        print("\nAverages: " + ", ".join(f"{label} {sum(getattr(r[2], k) for r in scored) / n:.1f}" for k, label in DIMS)
              + f" | total {sum(r[2].total for r in scored) / n:.1f}/50 over {n} messages")
        print("Lowest: " + ", ".join(f"{r[0]} ({r[2].total})" for r in sorted(scored, key=lambda r: r[2].total)[:6]))
    out = ROOT / ".cache" / "scores.md"
    out.write_text("\n".join(report), encoding="utf-8")
    print(f"Scoring cost: ${bot.llm_spent() - spent0:.4f} (total ${bot.llm_spent():.4f}). Details: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
