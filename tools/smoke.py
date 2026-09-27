"""In-process contract checks for bot.py (no server, no LLM).

Usage: python tools/smoke.py [--preview]
--preview also writes every test-pair composition to .cache/preview.md for manual review.
"""
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ["MOCK_LLM"] = "1"  # contract checks never spend credit (cached LLM results are still used)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from fastapi.testclient import TestClient  # noqa: E402

import bot  # noqa: E402
from load_dataset import load_dataset  # noqa: E402

ACTION_KEYS = {"conversation_id", "merchant_id", "customer_id", "send_as", "trigger_id", "template_name",
               "template_params", "body", "cta", "suppression_key", "rationale"}
CTAS = {"binary_yes_no", "binary_confirm_cancel", "multi_choice_slot", "open_ended", "none"}
LEAKS = [(r"\{\{|\}\}", "unrendered template slot"), (r"\bNone\b", "None"), (r"placeholder|metric_or_topic", "placeholder leak"),
         (r"https?://|www\.", "URL"), (r"\b[a-z0-9]+_[a-z0-9_]+\b", "snake_case jargon"), (r"\b\d{4}-\d{2}-\d{2}\b", "raw ISO date"),
         (r"\s[,.]", "space before punctuation"), (r"\b(\w+) \1\b", "doubled word")]

failures: list[str] = []


def check(ok: bool, label: str) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    if not ok:
        failures.append(label)


def lint_message(msg: dict, where: str) -> list[str]:
    problems = []
    body = msg.get("body") or ""
    if not body.strip():
        problems.append("empty body")
    for pattern, label in LEAKS:
        if re.search(pattern, body):
            problems.append(f"{label}: {re.search(pattern, body).group(0)!r}")
    if msg.get("cta") not in CTAS:
        problems.append(f"cta {msg.get('cta')!r}")
    if bot.render(bot.TEMPLATES[msg["template_name"]], msg["template_params"]) != body:
        problems.append("body != rendered template")
    return [f"{where}: {p}" for p in problems]


def main() -> None:
    preview = "--preview" in sys.argv
    client = TestClient(bot.app)
    data = load_dataset("expanded")
    ctx = lambda scope, v=1, **kw: {"scope": scope, "version": v, "delivered_at": "2026-04-26T09:45:00Z", **kw}  # noqa: E731

    print("\n1. health + metadata")
    r = client.get("/v1/healthz").json()
    check(r["status"] == "ok" and set(r["contexts_loaded"].values()) == {0}, "healthz starts empty")
    meta = client.get("/v1/metadata").json()
    check({"team_name", "team_members", "model", "approach", "contact_email", "version", "submitted_at"} <= set(meta),
          "metadata has all fields")

    print("\n2. warmup push (5 categories, 50 merchants, 200 customers)")
    for scope in ("category", "merchant", "customer"):
        codes = {client.post("/v1/context", json=ctx(scope, context_id=cid, payload=p)).status_code for cid, p in data[scope]}
        check(codes == {200}, f"all {scope} pushes accepted")
    check(client.get("/v1/healthz").json()["contexts_loaded"] == {"category": 5, "merchant": 50, "customer": 200, "trigger": 0},
          "healthz counts = 255")

    print("\n3. versioning + validation")
    mid, m = data["merchant"][0]
    r = client.post("/v1/context", json=ctx("merchant", context_id=mid, payload=m))
    check(r.status_code == 409 and r.json() == {"accepted": False, "reason": "stale_version", "current_version": 1},
          "same version -> 409 stale_version")
    check(client.post("/v1/context", json=ctx("merchant", 0, context_id=mid, payload=m)).status_code == 409, "lower version -> 409")
    m2 = json.loads(json.dumps(m))
    m2["performance"]["views"] = 2580
    r = client.post("/v1/context", json=ctx("merchant", 2, context_id=mid, payload=m2))
    check(r.status_code == 200 and bot.store.find("merchant", mid)["performance"]["views"] == 2580, "higher version replaces")
    r = client.post("/v1/context", json=ctx("merchnt", context_id="x", payload={}))
    check(r.status_code == 400 and r.json()["reason"] == "invalid_scope", "bad scope -> 400 invalid_scope")
    check(client.post("/v1/context", json={"scope": "merchant", "context_id": "x", "payload": {}}).status_code == 400,
          "missing version -> 400")
    check(client.post("/v1/context", content=b"not json", headers={"Content-Type": "application/json"}).status_code == 400,
          "non-JSON -> 400")
    check(client.get("/v1/healthz").json()["contexts_loaded"]["merchant"] == 50, "counts unchanged by rejected pushes")

    print("\n4. compose() over all 100 triggers")
    customers = dict(data["customer"])
    problems, t0 = [], time.time()
    for tid, trg in data["trigger"]:
        merchant = bot.store.find("merchant", trg["merchant_id"])
        category = bot.store.find("category", merchant["category_slug"])
        customer = customers.get(trg.get("customer_id"))
        msg = bot.compose(category, merchant, trg, customer)
        problems += lint_message(msg, tid)
        expected = "merchant_on_behalf" if trg["scope"] == "customer" else "vera"
        if msg["send_as"] != expected:
            problems.append(f"{tid}: send_as {msg['send_as']}")
    for p in problems[:15]:
        print("     ", p)
    check(not problems, f"100 compositions clean ({len(problems)} problems, {time.time() - t0:.2f}s)")

    print("\n5. tick policy")
    for tid, trg in data["trigger"]:
        client.post("/v1/context", json=ctx("trigger", context_id=tid, payload=trg))
    all_ids = [tid for tid, _ in data["trigger"]]
    t0 = time.time()
    acts = client.post("/v1/tick", json={"now": "2026-04-26T10:30:00Z", "available_triggers": all_ids}).json()["actions"]
    elapsed = time.time() - t0
    check(0 < len(acts) <= 20, f"tick returns 1..20 actions (got {len(acts)})")
    check(elapsed < 1.0, f"tick under 1s ({elapsed:.3f}s)")
    check(all(set(a) == ACTION_KEYS for a in acts), "every action has exactly the 11 required fields")
    check(all(a["body"].strip() for a in acts), "no empty bodies")
    vera_merchants = [a["merchant_id"] for a in acts if a["send_as"] == "vera"]
    check(len(vera_merchants) == len(set(vera_merchants)), "at most one merchant-facing action per merchant")
    cust = [a["customer_id"] for a in acts if a["customer_id"]]
    check(len(cust) == len(set(cust)), "at most one action per customer")
    check(len({a["conversation_id"] for a in acts}) == len(acts), "conversation_ids unique")
    by_id = dict(data["trigger"])
    urg = [int(by_id[a["trigger_id"]]["urgency"]) for a in acts]
    check(urg == sorted(urg, reverse=True), "actions ordered by urgency")
    first_sup = {a["suppression_key"] for a in acts}
    acts2 = client.post("/v1/tick", json={"now": "2026-04-26T10:35:00Z", "available_triggers": all_ids}).json()["actions"]
    check(not first_sup & {a["suppression_key"] for a in acts2}, "no suppression key sent twice")
    recent = set(vera_merchants)
    repeat = [a for a in acts2 if a["send_as"] == "vera" and a["merchant_id"] in recent and int(by_id[a["trigger_id"]]["urgency"]) < 4]
    check(not repeat, "cooldown: no non-urgent re-message within 20 min")
    check(all(a["customer_id"] != "c_015_anonymous_for_m010" for a in acts + acts2), "never messages customer without consent")

    print("\n6. reply plumbing")
    conv = acts[0]["conversation_id"]
    r = client.post("/v1/reply", json={"conversation_id": conv, "merchant_id": acts[0]["merchant_id"], "from_role": "merchant",
                                        "message": "ok", "received_at": "2026-04-26T10:40:00Z", "turn_number": 2}).json()
    check(r.get("action") in {"send", "wait", "end"} and "rationale" in r, "reply returns a valid action")
    r = client.post("/v1/reply", json={"conversation_id": "conv_never_seen", "from_role": "merchant", "message": "hello",
                                        "received_at": "2026-04-26T10:41:00Z", "turn_number": 2})
    check(r.status_code == 200 and r.json().get("action") in {"send", "wait", "end"}, "unknown conversation, no merchant_id -> 200")
    bot.store.conversations[conv]["status"] = "ended"
    r = client.post("/v1/reply", json={"conversation_id": conv, "from_role": "merchant", "message": "hi again",
                                        "received_at": "2026-04-26T10:42:00Z", "turn_number": 3}).json()
    check(r["action"] == "end", "ended conversation stays ended")

    print("\n7. teardown")
    client.post("/v1/teardown")
    check(set(client.get("/v1/healthz").json()["contexts_loaded"].values()) == {0}, "teardown wipes state")

    if preview:
        write_preview(data)
    print(f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} FAILED: ' + '; '.join(failures)}")
    sys.exit(1 if failures else 0)


def write_preview(data) -> None:
    pairs = json.loads((ROOT / "dataset/expanded/test_pairs.json").read_text(encoding="utf-8"))["pairs"]
    cats = dict(data["category"])
    merchants, customers, triggers = dict(data["merchant"]), dict(data["customer"]), dict(data["trigger"])
    lines = ["# Fallback template output for the 30 test pairs\n"]
    for p in pairs:
        trg, merchant = triggers[p["trigger_id"]], merchants[p["merchant_id"]]
        msg = bot.compose(cats[merchant["category_slug"]], merchant, trg, customers.get(p["customer_id"]))
        tag = " (placeholder)" if trg["payload"].get("placeholder") else ""
        lines.append(f"## {p['test_id']} {trg['kind']}{tag} | {msg['send_as']} | {msg['cta']} | {msg['template_name']}\n\n"
                     f"{msg['body']}\n\n_rationale: {msg['rationale']}_\n")
    out = ROOT / ".cache" / "preview.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\npreview written to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
