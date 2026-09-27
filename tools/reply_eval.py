"""Live multi-turn conversations with LLM-written replies: prints transcripts, who wrote each reply and the cost.

Usage: python tools/reply_eval.py [--mock]    (--mock: cache only, no new API calls)
Replies marked [llm] were written by the LLM; [rule] means the rule text was kept (LLM off, failed or rejected).
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

QUALIFYING = ["would you", "do you", "can you tell", "what if", "how about"]
ACTIONING = ["done", "sending", "draft", "here", "confirm", "proceed", "next"]

SCENARIOS = [
    ("Research digest: question, commit, confirm", "trg_001_research_digest_dentists", "merchant",
     ["Interesting. Does this apply to my patients with implants?", "ok let's do it", "confirm"]),
    ("Perf dip: why, then go ahead", "trg_004_perf_dip_bharat", "merchant",
     ["Why did my calls drop so much?", "go ahead"]),
    ("Planning: engaged detail, then yes", "trg_013_corporate_thali_planning", "merchant",
     ["Looks good, but I want to keep it for bigger office orders only", "yes"]),
    ("Customer recall: Hinglish question, then slot", "trg_003_recall_due_priya", "customer",
     ["Saturday possible hai kya?", "2"]),
    ("Hinglish owner: engaged, then commit", "trg_008_curious_ask_studio11", "merchant",
     ["Is hafte bridal trial ki bahut demand hai", "haan karo"]),
    ("Supply alert: question the facts can't answer", "trg_018_supply_atorvastatin_recall", "merchant",
     ["How many of my customers got these batches?"]),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()
    if args.mock:
        os.environ["MOCK_LLM"] = "1"

    from fastapi.testclient import TestClient
    import bot
    from load_dataset import load_dataset

    client = TestClient(bot.app)
    data = load_dataset("expanded")

    def fresh():
        client.post("/v1/teardown")
        for scope in ("category", "merchant", "customer", "trigger"):
            for cid, payload in data[scope]:
                client.post("/v1/context", json={"scope": scope, "context_id": cid, "version": 1, "payload": payload,
                                                 "delivered_at": "2026-04-26T09:00:00Z"})

    def reply(conv, msg, role, turn, merchant=None):
        before = len(bot.REPLY_REJECTIONS)
        body = {"conversation_id": conv, "from_role": role, "message": msg, "turn_number": turn,
                "received_at": f"2026-04-26T10:{10 + turn:02d}:00Z"}
        if merchant:
            body["merchant_id"] = merchant
        r = client.post("/v1/reply", json=body).json()
        src = "llm" if "LLM-written" in r.get("rationale", "") else "rule"
        text = r.get("body") or (f"(wait {r['wait_seconds']}s)" if r["action"] == "wait" else "")
        print(f"  {role}: {msg}\n    -> {r['action']} [{src}] {r.get('cta', '')}\n       " + str(text).replace("\n", "\n       "))
        for rej in bot.REPLY_REJECTIONS[before:]:
            print(f"       (LLM draft rejected: {rej['problems']})")
        return r

    spent0 = bot.llm_spent()
    for title, tid, role, turns in SCENARIOS:
        print(f"\n=== {title}")
        fresh()
        act = client.post("/v1/tick", json={"now": "2026-04-26T10:00:00Z", "available_triggers": [tid]}).json()["actions"][0]
        print(f"  opener: {act['body']}")
        for i, msg in enumerate(turns, start=2):
            reply(act["conversation_id"], msg, role, i)

    print("\n=== Local-judge intent check (no trigger context)")
    fresh()
    r = reply("conv_intent_eval", "Ok lets do it. Whats next?", "merchant", 2, merchant="m_001_drmeera_dentist_delhi")
    low = r.get("body", "").lower()
    ok = any(w in low for w in ACTIONING) and not any(q in low for q in QUALIFYING)
    print(f"  judge heuristic: {'PASS' if ok else 'FAIL'}")

    print(f"\nThis run: ${bot.llm_spent() - spent0:.4f}. Total logged spend: ${bot.llm_spent():.4f}")


if __name__ == "__main__":
    main()
