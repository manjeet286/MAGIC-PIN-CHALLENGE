"""Simulated 60-minute judge test window against bot.py in-process.

Mimics the official harness: warmup (255 contexts), 12 ticks 5 simulated minutes apart with triggers released over
time, mid-test injections (new digest item + trigger, merchant perf update + trigger, new customer then recall_due),
and scripted merchant/customer personas replying for up to 3 turns (no LLM plays the merchant).

Checks: latency, response schemas, restraint, repeats, opt-out respect, and whether new context is used.
Usage: python tools/sim_window.py [--mock]   (--mock: no new API calls)
"""
import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

ACTION_KEYS = {"conversation_id", "merchant_id", "customer_id", "send_as", "trigger_id", "template_name",
               "template_params", "body", "cta", "suppression_key", "rationale"}
SHAPES = {"send": {"action", "body", "cta", "rationale"}, "wait": {"action", "wait_seconds", "rationale"},
          "end": {"action", "rationale"}}
MERCHANT_PERSONAS = {
    "engaged": ["Yes please, go ahead", "confirm"],
    "question": ["How much would this cost me?", "ok do it"],
    "auto_reply": ["Thank you for contacting us! Our team will respond shortly."] * 3,
    "hard_no": ["Not interested. Stop messaging me."],
    "silent": [],
    "hinglish": ["haan theek hai, bhej do", "confirm"],
    "busy": ["busy right now, message me tomorrow"],
}
CUSTOMER_PERSONAS = {"books": ["1"], "asks": ["Is Saturday possible?", "2"], "stops": ["STOP"], "silent": []}
T0 = datetime(2026, 4, 26, 10, 0, tzinfo=timezone.utc)


def iso(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


def pick(options: dict, key: str) -> str:
    names = sorted(options)
    return names[int(hashlib.sha1(key.encode()).hexdigest(), 16) % len(names)]


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
    merchants = dict(data["merchant"])
    triggers = dict(data["trigger"])
    pairs = json.loads((ROOT / "dataset/expanded/test_pairs.json").read_text(encoding="utf-8"))["pairs"]
    failures, latencies = [], {"tick": [], "reply": [], "context": []}
    sent_by_merchant, bodies_by_conv, opted_out, to_owner = {}, {}, {}, {}
    versions = {}

    def check(ok, label):
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        if not ok:
            failures.append(label)

    def push(scope, cid, payload):
        versions[cid] = versions.get(cid, 0) + 1
        t = time.time()
        r = client.post("/v1/context", json={"scope": scope, "context_id": cid, "version": versions[cid], "payload": payload,
                                             "delivered_at": iso(T0)})
        latencies["context"].append(time.time() - t)
        if r.status_code != 200:
            failures.append(f"context {scope}/{cid} -> {r.status_code}")

    spent0 = bot.llm_spent()
    client.post("/v1/teardown")
    print("\nWarmup")
    for scope in ("category", "merchant", "customer"):
        for cid, payload in data[scope]:
            push(scope, cid, payload)
    check(client.get("/v1/healthz").json()["contexts_loaded"] == {"category": 5, "merchant": 50, "customer": 200, "trigger": 0},
          "healthz shows exactly 255 contexts")

    # Trigger schedule: the 30 test-pair triggers plus the remaining seed triggers, released ~3-4 per tick
    seed_ids = [tid for tid, t in data["trigger"] if not t["payload"].get("placeholder")]
    release = list(dict.fromkeys([p["trigger_id"] for p in pairs] + seed_ids))
    schedule = {i: release[i * 4:(i + 1) * 4] for i in range(12)}

    # Injection targets: merchants with no scheduled trigger, so restraint rules don't hide the adaptation checks
    busy = {triggers[t]["merchant_id"] for t in release}
    idle_dentist = next(mid for mid, m in merchants.items() if m["category_slug"] == "dentists" and mid not in busy)
    idle_other = next(mid for mid, m in merchants.items() if m["category_slug"] == "salons" and mid not in busy)
    new_item = {"id": "d_2026W18_sim_sdf", "kind": "research",
                "title": "Silver diamine fluoride arrests most early caries lesions in a 12-month school trial",
                "source": "IJDR May 2026, p.22", "trial_n": 640, "patient_segment": "children",
                "summary": "Twice-yearly SDF arrested 81% of early lesions vs 44% with fluoride varnish alone in 640 children.",
                "actionable": "Consider SDF for pediatric patients with early lesions who can't sit for restorations"}
    new_customer = {"customer_id": "c_sim_ananya", "merchant_id": idle_other,
                    "identity": {"name": "Ananya", "phone_redacted": "<phone>", "language_pref": "english", "age_band": "25-35"},
                    "relationship": {"first_visit": "2025-10-02", "last_visit": "2026-01-20", "visits_total": 3,
                                     "services_received": ["hair_spa", "haircut", "hair_spa"]},
                    "state": "lapsed_soft",
                    "preferences": {"preferred_slots": "saturday_afternoon", "channel": "whatsapp", "reminder_opt_in": True},
                    "consent": {"opted_in_at": "2025-10-02", "scope": ["recall_reminders", "appointment_reminders"]}}
    injected = {}

    def converse(action, now, tick_index):
        conv, role = action["conversation_id"], "customer" if action["customer_id"] else "merchant"
        personas = CUSTOMER_PERSONAS if role == "customer" else MERCHANT_PERSONAS
        persona = pick(personas, conv)
        script = personas[persona]
        for turn, msg in enumerate(script, start=2):
            t = time.time()
            r = client.post("/v1/reply", json={"conversation_id": conv, "merchant_id": action["merchant_id"],
                                                "customer_id": action["customer_id"], "from_role": role, "message": msg,
                                                "received_at": iso(now + timedelta(minutes=turn)), "turn_number": turn})
            latencies["reply"].append(time.time() - t)
            out = r.json()
            if r.status_code != 200 or set(out) != SHAPES.get(out.get("action"), set()):
                failures.append(f"reply schema {conv}: {out}")
            if out.get("action") == "send":
                bodies_by_conv.setdefault(conv, []).append(out["body"])
            if out.get("action") == "end":
                if role == "merchant" and persona == "hard_no":
                    opted_out[action["merchant_id"]] = tick_index
                break
            if out.get("action") == "wait":
                break

    print("\nTest window: 12 ticks x 5 simulated minutes")
    released = []
    for i in range(12):
        now = T0 + timedelta(minutes=5 * i)
        if i == 3:
            cat = json.loads(json.dumps(dict(data["category"])["dentists"]))
            cat["digest"].append(new_item)
            push("category", "dentists", cat)
            trg = {"id": "trg_sim_research_w18", "scope": "merchant", "kind": "research_digest", "source": "external",
                   "merchant_id": idle_dentist, "customer_id": None,
                   "payload": {"category": "dentists", "top_item_id": new_item["id"]}, "urgency": 3,
                   "suppression_key": "research:dentists:2026-W18", "expires_at": "2026-05-10T00:00:00Z"}
            push("trigger", trg["id"], trg)
            released.append(trg["id"])
        if i == 5:
            m = json.loads(json.dumps(merchants[idle_other]))
            m["performance"].update({"views": 3333, "calls": 44, "delta_7d": {"views_pct": -0.21, "calls_pct": -0.37}})
            injected["perf"] = (m["performance"], merchants[idle_other]["performance"])
            push("merchant", idle_other, m)
            trg = {"id": "trg_sim_perf_dip", "scope": "merchant", "kind": "perf_dip", "source": "internal",
                   "merchant_id": idle_other, "customer_id": None,
                   "payload": {"metric": "calls", "delta_pct": -0.37, "window": "7d", "vs_baseline": 70}, "urgency": 4,
                   "suppression_key": "perf_dip:sim:calls", "expires_at": "2026-05-10T00:00:00Z"}
            push("trigger", trg["id"], trg)
            released.append(trg["id"])
        if i == 7:
            push("customer", new_customer["customer_id"], new_customer)
        if i == 8:
            trg = {"id": "trg_sim_recall_ananya", "scope": "customer", "kind": "recall_due", "source": "internal",
                   "merchant_id": idle_other, "customer_id": new_customer["customer_id"],
                   "payload": {"service_due": "hair_spa_followup", "last_service_date": "2026-01-20",
                               "available_slots": [{"iso": "2026-05-02T15:00:00+05:30", "label": "Sat 2 May, 3pm"},
                                                   {"iso": "2026-05-09T15:00:00+05:30", "label": "Sat 9 May, 3pm"}]},
                   "urgency": 3, "suppression_key": "recall:c_sim_ananya", "expires_at": "2026-05-30T00:00:00Z"}
            push("trigger", trg["id"], trg)
            released.append(trg["id"])
        for tid in schedule[i]:
            push("trigger", tid, triggers[tid])
            released.append(tid)
        t = time.time()
        r = client.post("/v1/tick", json={"now": iso(now), "available_triggers": released})
        latencies["tick"].append(time.time() - t)
        acts = r.json().get("actions", [])
        bad = [a for a in acts if set(a) != ACTION_KEYS or not a["body"].strip()]
        if r.status_code != 200 or bad:
            failures.append(f"tick {i} schema: {bad[:1]}")
        for a in acts:
            sent_by_merchant.setdefault(a["merchant_id"], []).append(i)
            if a["send_as"] == "vera":
                to_owner.setdefault(a["merchant_id"], []).append((i, a["trigger_id"]))
            bodies_by_conv.setdefault(a["conversation_id"], []).append(a["body"])
            injected.setdefault("acts", []).append(a)
        print(f"  tick {i:2} {iso(now)[11:16]}: {len(released):3} active triggers -> {len(acts)} actions "
              f"({latencies['tick'][-1]:.1f}s)")
        for a in acts:
            converse(a, now, i)

    acts = injected.get("acts", [])
    print("\nOperational")
    check(not [f for f in failures if "schema" in f or "context" in f], "all tick/reply/context responses valid")
    p95 = lambda xs: sorted(xs)[max(0, int(len(xs) * 0.95) - 1)] if xs else 0  # noqa: E731
    for kind, xs in latencies.items():
        print(f"     {kind:7} n={len(xs):3}  median={statistics.median(xs) if xs else 0:.2f}s  p95={p95(xs):.2f}s  max={max(xs, default=0):.2f}s")
    check(max(latencies["tick"] + latencies["reply"], default=0) < 10, "every tick and reply under 10s")
    print("\nRestraint and hygiene")
    counts = {m: len(v) for m, v in sent_by_merchant.items()}
    owner_counts = {m: len(v) for m, v in to_owner.items()}
    busiest = max(owner_counts, key=owner_counts.get, default=None)
    check(max(owner_counts.values(), default=0) <= 3,
          f"no owner gets more than 3 Vera conversations in 60 min (max {owner_counts.get(busiest, 0)}: {to_owner.get(busiest)})")
    check(all(len(b) == len(set(b)) for b in bodies_by_conv.values()), "no verbatim repeat inside any conversation")
    late = [m for m, t in opted_out.items() if any(ti > t for ti in sent_by_merchant.get(m, []))]
    check(not late, f"nothing sent to the {len(opted_out)} merchants after they opted out")
    print(f"     {len(acts)} conversations opened for {len(counts)} merchants over 12 ticks")
    print("\nAdaptation to injected context")
    research = [a for a in acts if a["trigger_id"] == "trg_sim_research_w18"]
    check(bool(research) and ("IJDR" in research[0]["body"] or "diamine" in research[0]["body"].lower()),
          "new digest item (pushed mid-test) is cited in the message")
    perf = [a for a in acts if a["trigger_id"] == "trg_sim_perf_dip"]
    new_p, old_p = injected["perf"]
    body = perf[0]["body"] if perf else ""
    check(bool(perf) and ("3,333" in body or "44" in body or "37%" in body), "updated performance numbers are used")
    check(bool(perf) and f"{old_p['views']:,}" not in body, "stale performance numbers are not used")
    recall = [a for a in acts if a["trigger_id"] == "trg_sim_recall_ananya"]
    check(bool(recall) and recall[0]["send_as"] == "merchant_on_behalf" and "Ananya" in recall[0]["body"],
          "new customer's recall is sent on the merchant's behalf, by name")
    for label, found in (("research", research), ("perf", perf), ("recall", recall)):
        if found:
            print(f"     {label}: {found[0]['body'][:230]}")

    print(f"\nLLM cost of this run: ${bot.llm_spent() - spent0:.4f} (total ${bot.llm_spent():.4f})")
    print("ALL SIM CHECKS PASSED" if not failures else f"{len(failures)} FAILED: " + "; ".join(failures[:6]))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
