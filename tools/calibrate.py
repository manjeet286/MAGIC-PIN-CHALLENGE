"""Calibrate the local judge: score the brief's 10 case-study messages (rated 44-50/50 in examples/case-studies.md)
and our own message for the same trigger, side by side, with the same judge.

Usage: python tools/calibrate.py     (judge calls are cached, so re-runs of unchanged messages are free)
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

# (trigger_id, cta, brief's own score) for case studies 1-10, in document order
CASES = [("trg_001_research_digest_dentists", "open_ended", 50), ("trg_003_recall_due_priya", "multi_choice_slot", 49),
         ("trg_007_bridal_followup_kavya", "binary_yes_no", 47), ("trg_008_curious_ask_studio11", "open_ended", 44),
         ("trg_010_ipl_match_delhi", "binary_yes_no", 50), ("trg_013_corporate_thali_planning", "binary_yes_no", 49),
         ("trg_014_seasonal_acquisition_dip_powerhouse", "binary_yes_no", 48), ("trg_015_winback_rashmi", "binary_yes_no", 50),
         ("trg_018_supply_atorvastatin_recall", "binary_yes_no", 50), ("trg_019_chronic_refill_grandfather", "binary_confirm_cancel", 49)]


def main() -> None:
    import bot
    from load_dataset import load_dataset
    from score_submission import DIMS, make_scorer

    data = load_dataset("expanded")
    cats, merchants = dict(data["category"]), dict(data["merchant"])
    customers, triggers = dict(data["customer"]), dict(data["trigger"])
    text = (ROOT / "examples" / "case-studies.md").read_text(encoding="utf-8")
    bodies = [" ".join(b.split()) for b in re.findall(r"```\n(.*?)\n```", text, re.S)]
    assert len(bodies) == len(CASES), f"expected {len(CASES)} case-study messages, found {len(bodies)}"

    scorer = make_scorer()
    spent0 = bot.llm_spent()
    rows = []
    print(f"{'trigger':34} {'brief':>5} | {'case study (judge)':>22} | {'ours (judge)':>22}")
    for (tid, cta, brief_score), cs_body in zip(CASES, bodies):
        trg = triggers[tid]
        m = merchants[trg["merchant_id"]]
        cat, cust = cats[m["category_slug"]], customers.get(trg.get("customer_id"))
        send_as = "merchant_on_behalf" if trg.get("scope") == "customer" else "vera"
        cs = scorer.score({"body": cs_body, "cta": cta, "send_as": send_as}, cat, m, trg, cust)
        ours_msg = bot.compose(cat, m, trg, cust)
        ours = scorer.score(ours_msg, cat, m, trg, cust)
        rows.append((tid, brief_score, cs, ours, ours_msg["body"]))
        fmt = lambda s: " ".join(f"{getattr(s, k)}" for k, _ in DIMS) + f" ={s.total:>3}"  # noqa: E731
        print(f"{tid[:34]:34} {brief_score:>5} | {fmt(cs):>22} | {fmt(ours):>22}")

    n = len(rows)
    for label, idx in (("case studies", 2), ("ours", 3)):
        avg = {k: sum(getattr(r[idx], k) for r in rows) / n for k, _ in DIMS}
        print(f"{label:>12}: " + ", ".join(f"{lab} {avg[k]:.1f}" for k, lab in DIMS)
              + f" | total {sum(r[idx].total for r in rows) / n:.1f}/50")
    print(f"brief's own scores for the case studies average {sum(r[1] for r in rows) / n:.1f}/50")
    print(f"Calibration cost: ${bot.llm_spent() - spent0:.4f} (total ${bot.llm_spent():.4f})")


if __name__ == "__main__":
    main()
