"""Generate submission.jsonl for the 30 canonical test pairs (brief §7.2) and lint every line.

Usage: python tools/make_submission.py [--mock]   (--mock: no new API calls; cached LLM output, else templates)
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

KEYS = ["test_id", "body", "cta", "send_as", "suppression_key", "rationale"]
LINT = [(r"\{\{|\}\}", "unrendered slot"), (r"\bNone\b", "None"), (r"https?://|www\.", "URL"),
        (r"\b[a-z0-9]+_[a-z0-9_]+\b", "snake_case"), (r"placeholder|payload|suppression", "internal term")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()
    if args.mock:
        os.environ["MOCK_LLM"] = "1"

    import bot
    from load_dataset import load_dataset

    data = load_dataset("expanded")
    cats, merchants = dict(data["category"]), dict(data["merchant"])
    customers, triggers = dict(data["customer"]), dict(data["trigger"])
    pairs = json.loads((ROOT / "dataset/expanded/test_pairs.json").read_text(encoding="utf-8"))["pairs"]

    spent0 = bot.llm_spent()
    lines, problems, sources = [], [], {"llm": 0, "template": 0}
    for p in pairs:
        m = merchants[p["merchant_id"]]
        trg, cust = triggers[p["trigger_id"]], customers.get(p["customer_id"])
        msg = bot.compose(cats[m["category_slug"]], m, trg, cust)
        sources[msg["composer"]] += 1
        line = {"test_id": p["test_id"], **{k: msg[k] for k in KEYS[1:]}}
        lines.append(line)
        for pattern, label in LINT:
            hit = re.search(pattern, line["body"])
            if hit:
                problems.append(f"{p['test_id']}: {label} {hit.group(0)!r}")
        if line["cta"] not in bot.CTAS:
            problems.append(f"{p['test_id']}: cta {line['cta']!r}")
        expected = "merchant_on_behalf" if trg.get("scope") == "customer" else "vera"
        if line["send_as"] != expected:
            problems.append(f"{p['test_id']}: send_as {line['send_as']}")
        if not all(isinstance(line[k], str) and line[k].strip() for k in KEYS):
            problems.append(f"{p['test_id']}: empty field")

    out = ROOT / "submission.jsonl"
    out.write_text("\n".join(json.dumps(l, ensure_ascii=False) for l in lines) + "\n", encoding="utf-8")
    for pr in problems:
        print("  LINT:", pr)
    print(f"Wrote {len(lines)} lines to {out.name}: {sources['llm']} LLM-composed, {sources['template']} template. "
          f"Lint problems: {len(problems)}. Cost: ${bot.llm_spent() - spent0:.4f} (total ${bot.llm_spent():.4f})")
    sys.exit(1 if problems or len(lines) != 30 else 0)


if __name__ == "__main__":
    main()
