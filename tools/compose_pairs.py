"""Compose test pairs with the LLM composer and show what it produced, what it cost and why drafts were rejected.

Usage:
    python tools/compose_pairs.py                 # the 8-pair tuning set
    python tools/compose_pairs.py --pairs T21,T28 # specific pairs
    python tools/compose_pairs.py --all           # all 30 test pairs
    python tools/compose_pairs.py --mock          # no API calls (cache + templates only)
    python tools/compose_pairs.py --facts T28     # print the fact sheet sent to the LLM for one pair
Results go to .cache/compose_preview.md. Unchanged prompts are served from the cache for free.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

TUNING_SET = ["T21", "T01", "T30", "T28", "T07", "T13", "T25", "T03"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", help="comma-separated test ids")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--facts", help="print the fact sheet for one test id and exit")
    args = ap.parse_args()
    if args.mock:
        os.environ["MOCK_LLM"] = "1"

    import bot
    from load_dataset import load_dataset

    data = load_dataset("expanded")
    cats, merchants = dict(data["category"]), dict(data["merchant"])
    customers, triggers = dict(data["customer"]), dict(data["trigger"])
    pairs = json.loads((ROOT / "dataset/expanded/test_pairs.json").read_text(encoding="utf-8"))["pairs"]
    by_id = {p["test_id"]: p for p in pairs}

    def contexts(p):
        m = merchants[p["merchant_id"]]
        return cats[m["category_slug"]], m, triggers[p["trigger_id"]], customers.get(p["customer_id"])

    if args.facts:
        cat, m, trg, cust = contexts(by_id[args.facts])
        base = bot.compose_template(cat, m, trg, cust)
        facts = bot.build_facts(bot.Ctx(cat, m, trg, cust), base["send_as"] == "merchant_on_behalf", base)
        text = json.dumps(facts, ensure_ascii=False, separators=(",", ":"))
        print(json.dumps(facts, ensure_ascii=False, indent=1))
        print(f"\n~{len(text) // 4} tokens of facts + ~{len(bot.SYSTEM_PROMPT) // 4} tokens of system prompt")
        return

    ids = [p["test_id"] for p in pairs] if args.all else (args.pairs.split(",") if args.pairs else TUNING_SET)
    spent_before = bot.llm_spent()
    lines, n_llm = ["# LLM composer output\n"], 0
    for tid in ids:
        p = by_id[tid]
        cat, m, trg, cust = contexts(p)
        t0 = time.time()
        msg = bot.compose(cat, m, trg, cust)
        dt = time.time() - t0
        n_llm += msg["composer"] == "llm"
        tag = " (placeholder)" if (trg.get("payload") or {}).get("placeholder") else ""
        head = f"{tid} {trg['kind']}{tag} | {msg['composer']} | {msg['cta']} | {len(msg['body'])} chars | {dt:.1f}s"
        print(f"\n== {head}\n{msg['body']}")
        if msg.get("llm_problems"):
            print(f"   rejected: {msg['llm_problems']}")
        lines.append(f"## {head}\n\n{msg['body']}\n\n_rationale: {msg['rationale']}_\n")
        if msg.get("llm_problems"):
            lines.append(f"_LLM draft rejected: {'; '.join(msg['llm_problems'])}_\n")

    out = ROOT / ".cache" / "compose_preview.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    spent_after = bot.llm_spent()
    print(f"\n{n_llm}/{len(ids)} composed by the LLM. This run: ${spent_after - spent_before:.4f}. "
          f"Total logged spend: ${spent_after:.4f} (cap ${bot.SPEND_CAP:.2f}). Preview: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
