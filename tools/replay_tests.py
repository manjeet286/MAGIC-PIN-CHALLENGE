"""Scripted multi-turn conversations against bot.py in-process (no server, no LLM).

Covers the brief's replay scenarios (auto-reply hell, intent transition, hostile/off-topic), the
api-call-examples reply cases (2.4-2.7), Hinglish turns, customer booking flows and tick restraint.
Usage: python tools/replay_tests.py [-v]   (-v prints every bot turn)
"""
import os
import re
import sys
from pathlib import Path

os.environ["MOCK_LLM"] = "1"  # conversation tests never spend credit
os.environ["VERA_DISABLE_LLM"] = "1"  # and ignore the cache, so they test the deterministic rule layer
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from fastapi.testclient import TestClient  # noqa: E402

import bot  # noqa: E402
from load_dataset import load_dataset  # noqa: E402

VERBOSE = "-v" in sys.argv
QUALIFYING = ["would you", "do you", "can you tell", "what if", "how about"]
ACTIONING = ["done", "sending", "draft", "here", "confirm", "proceed", "next"]
CANNED = "Thank you for contacting Dr. Meera's Dental Clinic! Our team will respond shortly."

client = TestClient(bot.app)
DATA = load_dataset("expanded")
failures: list[str] = []
all_responses: list[dict] = []


def check(ok: bool, label: str) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    if not ok:
        failures.append(label)


def fresh() -> None:
    client.post("/v1/teardown")
    for scope in ("category", "merchant", "customer", "trigger"):
        for cid, payload in DATA[scope]:
            client.post("/v1/context", json={"scope": scope, "context_id": cid, "version": 1, "payload": payload,
                                             "delivered_at": "2026-04-26T09:00:00Z"})


def tick(now: str, triggers: list[str]) -> list[dict]:
    acts = client.post("/v1/tick", json={"now": now, "available_triggers": triggers}).json()["actions"]
    if VERBOSE:
        for a in acts:
            print(f"      tick -> {a['conversation_id']}: {a['body'][:110]}")
    return acts


def say(conv: str, msg: str, at: str, merchant: str | None = None, role: str = "merchant", turn: int = 2) -> dict:
    body = {"conversation_id": conv, "from_role": role, "message": msg, "received_at": at, "turn_number": turn}
    if merchant:
        body["merchant_id"] = merchant
    r = client.post("/v1/reply", json=body)
    out = r.json()
    all_responses.append(out)
    if VERBOSE:
        print(f"      {role}: {msg[:70]!r}\n        -> {out['action']}: {out.get('body', out.get('wait_seconds', ''))!s:.140}")
    return out


def conv_for(acts: list[dict], trigger_id: str) -> str:
    return next(a["conversation_id"] for a in acts if a["trigger_id"] == trigger_id)


def main() -> None:
    print("\nA. Auto-reply hell (same conversation, 4 identical canned replies)")
    fresh()
    acts = tick("2026-04-26T10:00:00Z", ["trg_022_cde_webinar_dentists"])
    conv = conv_for(acts, "trg_022_cde_webinar_dentists")
    r = [say(conv, CANNED, f"2026-04-26T10:0{i}:00Z", turn=i + 1) for i in range(1, 5)]
    check([x["action"] for x in r] == ["send", "wait", "end", "end"], f"send -> wait -> end -> end (got {[x['action'] for x in r]})")
    check("YES" in r[0].get("body", ""), "first response flags the auto-reply with a YES prompt")
    check(r[1].get("wait_seconds") == 86400, "second auto-reply waits 24h")
    check(not tick("2026-04-26T10:40:00Z", ["trg_001_research_digest_dentists"]), "merchant paused for new ticks after auto-replies")

    print("\nB. Auto-reply across fresh conversation ids (local-judge style)")
    fresh()
    r = [say(f"conv_auto_{i}", "Thank you for contacting us! Our team will respond shortly.", f"2026-04-26T10:0{i}:00Z",
             merchant="m_001_drmeera_dentist_delhi", turn=i + 1) for i in range(1, 5)]
    check([x["action"] for x in r][:3] == ["send", "wait", "end"], f"detected per merchant: {[x['action'] for x in r]}")

    print("\nC. Hindi auto-reply (Pattern B from the brief)")
    fresh()
    acts = tick("2026-04-26T10:00:00Z", ["trg_008_curious_ask_studio11"])
    r = say(conv_for(acts, "trg_008_curious_ask_studio11"),
            "Aapki jaankari ke liye bahut-bahut shukriya. Main aapki yeh sabhi baatein aur sujhaav hamari team tak pahuncha deti hoon.",
            "2026-04-26T10:02:00Z")
    check(r["action"] == "send" and "auto-reply" in r["body"].lower() and "YES" in r["body"], "detected, one Hinglish prompt")

    print("\nD. Intent transition (qualify, qualify, commit, confirm)")
    fresh()
    acts = tick("2026-04-26T11:00:00Z", ["trg_001_research_digest_dentists"])
    conv = conv_for(acts, "trg_001_research_digest_dentists")
    r1 = say(conv, "Tell me more about this", "2026-04-26T11:02:00Z", turn=2)
    r2 = say(conv, "What would the patient message say?", "2026-04-26T11:04:00Z", turn=3)
    r3 = say(conv, "Ok, let's do it. What's next?", "2026-04-26T11:06:00Z", turn=4)
    low = r3.get("body", "").lower()
    check(r1["action"] == "send" and r2["action"] == "send", "answers the qualifying turns")
    check(r3["action"] == "send" and any(w in low for w in ACTIONING) and not any(q in low for q in QUALIFYING),
          "commit switches to action mode (action words, no qualifying phrases)")
    r4 = say(conv, "confirm", "2026-04-26T11:08:00Z", turn=5)
    check(r4["action"] == "send" and r4["cta"] == "none", "confirmation closes the loop without a new ask")
    check(say(conv, "hello?", "2026-04-26T11:10:00Z", turn=6)["action"] == "end", "conversation closed after completion")

    print("\nE. Hostile, then off-topic (replay scenario 3)")
    fresh()
    acts = tick("2026-04-26T10:00:00Z", ["trg_004_perf_dip_bharat"])
    conv = conv_for(acts, "trg_004_perf_dip_bharat")
    check(say(conv, "Why are you bothering me. This is useless. Stop sending these.", "2026-04-26T10:02:00Z")["action"] == "end",
          "hostile -> end")
    check(say(conv, "can you also help me file my GST?", "2026-04-26T10:03:00Z", turn=3)["action"] == "end",
          "stays closed after hostility")
    check(not tick("2026-04-26T10:45:00Z", ["trg_005_renewal_due_bharat"]), "merchant suppressed even for urgent triggers")

    print("\nF. Hard no (example 2.6)")
    fresh()
    acts = tick("2026-04-26T10:00:00Z", ["trg_018_supply_atorvastatin_recall"])
    r = say(conv_for(acts, "trg_018_supply_atorvastatin_recall"), "Not interested. Stop messaging me.", "2026-04-26T10:02:00Z")
    check(r["action"] == "end", "opt-out -> end")
    check(not tick("2026-04-26T11:00:00Z", ["trg_020_summer_demand_shift"]), "no further outreach to that merchant")

    print("\nG. Curveball off-topic, polite (example 2.7)")
    fresh()
    acts = tick("2026-04-26T10:00:00Z", ["trg_021_unverified_gbp_sunrise"])
    conv = conv_for(acts, "trg_021_unverified_gbp_sunrise")
    r1 = say(conv, "Btw can you also help me with my GST filing this month?", "2026-04-26T10:02:00Z")
    r2 = say(conv, "And my income tax return?", "2026-04-26T10:03:00Z", turn=3)
    check(r1["action"] == "send" and "CA" in r1["body"] and "YES" in r1["body"], "declines GST, redirects with one CTA")
    check(r2["action"] == "send" and r2["body"] != r1["body"], "second off-topic reply is not a repeat")

    print("\nH. Defer")
    fresh()
    acts = tick("2026-04-26T10:00:00Z", ["trg_012_milestone_mylari"])
    r = say(conv_for(acts, "trg_012_milestone_mylari"), "Busy right now, call me tomorrow", "2026-04-26T10:02:00Z")
    check(r["action"] == "wait" and r["wait_seconds"] == 86400, "busy + tomorrow -> wait 24h")

    print("\nI. Hinglish turns")
    fresh()
    acts = tick("2026-04-26T10:00:00Z", ["trg_016_kids_yoga_program_drafting"])
    conv = conv_for(acts, "trg_016_kids_yoga_program_drafting")
    r1 = say(conv, "haan karo, jaldi bhejo", "2026-04-26T10:02:00Z")
    check(r1["action"] == "send" and "CONFIRM" in r1["body"] and "kar rahi" in r1["body"], "Hinglish commit -> Hinglish action")
    check(say(conv, "baad mein baat karte hain", "2026-04-26T10:04:00Z", turn=3)["action"] == "wait", "Hinglish defer -> wait")

    print("\nJ. Engaged accept (example 2.4)")
    fresh()
    acts = tick("2026-04-26T10:00:00Z", ["trg_001_research_digest_dentists"])
    r = say(conv_for(acts, "trg_001_research_digest_dentists"), "Yes please send the abstract. Also draft the patient WhatsApp.",
            "2026-04-26T10:02:00Z")
    check(r["action"] == "send" and "draft" in r["body"].lower(), "accept -> action")

    print("\nK. Customer recall booking")
    fresh()
    acts = tick("2026-04-26T11:00:00Z", ["trg_003_recall_due_priya"])
    conv = conv_for(acts, "trg_003_recall_due_priya")
    check(acts[0]["send_as"] == "merchant_on_behalf", "recall sent on the merchant's behalf")
    r = say(conv, "2", "2026-04-26T11:05:00Z", role="customer")
    check(r["action"] == "send" and "Thu 6 Nov, 5pm" in r["body"], "slot '2' books Thu 6 Nov, 5pm")
    fresh()
    acts = tick("2026-04-26T11:00:00Z", ["trg_003_recall_due_priya"])
    conv = conv_for(acts, "trg_003_recall_due_priya")
    r = say(conv, "haan", "2026-04-26T11:05:00Z", role="customer")
    check(r["action"] == "send" and r["cta"] == "multi_choice_slot" and "Wed 5 Nov" in r["body"], "'haan' -> offers the real slots")
    check(say(conv, "STOP", "2026-04-26T11:06:00Z", role="customer", turn=3)["action"] == "end", "customer STOP -> end")
    check(bot.store.cstate("c_001_priya_for_m001")["opted_out"], "customer opted out permanently")

    print("\nL. Customer refill confirm (Hindi profile)")
    fresh()
    acts = tick("2026-04-26T11:00:00Z", ["trg_019_chronic_refill_grandfather"])
    r = say(conv_for(acts, "trg_019_chronic_refill_grandfather"), "haan bhej do", "2026-04-26T11:05:00Z", role="customer")
    check(r["action"] == "send" and "metformin" in r["body"] and "Confirm ho gaya" in r["body"], "refill confirmed in Hinglish")

    print("\nM. Restraint: stop after 3 unanswered nudges")
    fresh()
    m001 = ["trg_001_research_digest_dentists", "trg_002_compliance_dci_radiograph", "trg_022_cde_webinar_dentists",
            "trg_023_competitor_opened_dentist"]
    sent = [tick(t, m001) for t in ("2026-04-26T10:00:00Z", "2026-04-26T10:30:00Z", "2026-04-26T11:00:00Z", "2026-04-26T11:30:00Z")]
    check([len(s) for s in sent] == [1, 1, 1, 0], f"1,1,1 then silence (got {[len(s) for s in sent]})")
    check(sent[0] and sent[0][0]["trigger_id"] == "trg_002_compliance_dci_radiograph", "urgent compliance trigger goes first")
    say(sent[2][0]["conversation_id"], "ok tell me more", "2026-04-26T11:40:00Z")
    check(len(tick("2026-04-26T12:10:00Z", m001)) == 1, "a real reply resets the counter")

    print("\nN. Response invariants")
    shapes = {"send": {"action", "body", "cta", "rationale"}, "wait": {"action", "wait_seconds", "rationale"},
              "end": {"action", "rationale"}}
    check(all(set(r) == shapes[r["action"]] for r in all_responses), f"all {len(all_responses)} replies have exact schemas")
    check(all(r["body"].strip() for r in all_responses if r["action"] == "send"), "no empty sends")
    leaks = [r["body"] for r in all_responses if r["action"] == "send" and re.search(r"\{|\}|None|_[a-z]", r["body"])]
    check(not leaks, f"no template or jargon leaks {leaks[:1]}")

    print(f"\n{'ALL SCENARIOS PASSED' if not failures else f'{len(failures)} FAILED: ' + '; '.join(failures)}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
