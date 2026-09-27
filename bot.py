"""Vera: merchant-engagement bot for the magicpin AI Challenge.

Serves the judge harness over HTTP and exposes compose() for offline submission generation.
Run with a single worker (all state is in memory):
    uvicorn bot:app --host 0.0.0.0 --port 8080
"""
from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).with_name(".env"))
except ImportError:
    pass

import conversation_handlers

BOT_VERSION = "1.0.0"
STARTED = time.time()
MAX_ACTIONS_PER_TICK = 20
MAX_UNANSWERED = 3
URGENT = 4
# The judge's available_triggers is the source of truth for what is active; strict expiry is opt-in
# because the local judge sends wall-clock `now`, which makes most dataset triggers look expired.
STRICT_EXPIRY = os.environ.get("VERA_STRICT_EXPIRY") == "1"
COOLDOWN = timedelta(minutes=int(os.environ.get("VERA_COOLDOWN_MIN", "30")))
# Tick composes in parallel and must answer inside the judge's budget; unfinished LLM work falls back to templates.
TICK_BUDGET_S = float(os.environ.get("VERA_TICK_BUDGET_S", "9"))
EXECUTOR = ThreadPoolExecutor(max_workers=int(os.environ.get("VERA_LLM_WORKERS", "8")))
SCOPES = ("category", "merchant", "customer", "trigger")
ID_FIELD = {"category": "slug", "merchant": "merchant_id", "customer": "customer_id", "trigger": "id"}
FAR_FUTURE = datetime(9999, 1, 1, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def parse_ts(value) -> Optional[datetime]:
    if not value or not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def to_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def pct(x) -> str:
    return f"{round(abs(float(x)) * 100)}%"


def num(n) -> str:
    return f"{int(n):,}"


def inr(amount) -> str:
    s = str(int(round(float(amount))))
    if len(s) > 3:
        head = ",".join(re.findall(r"\d{1,2}", s[:-3][::-1]))[::-1]
        s = f"{head},{s[-3:]}"
    return f"₹{s}"


def human_date(value) -> str:
    if not value:
        return ""
    dt = parse_ts(value) if "T" in str(value) else None
    if dt is None:
        try:
            dt = datetime.strptime(str(value)[:10], "%Y-%m-%d")
        except ValueError:
            return str(value)
    return f"{dt.day} {dt:%b %Y}"


def human_time(dt: datetime) -> str:
    return f"{int(dt.strftime('%I'))}:{dt:%M} {dt.strftime('%p').lower()}"


def human_datetime(value) -> str:
    """'Sun 26 Apr, 7:30 pm'; midnight timestamps are really dates, so no time is shown."""
    dt = parse_ts(value)
    if not dt:
        return human_date(value)
    day = f"{dt:%a} {dt.day} {dt:%b}"
    return day + f" {dt.year}" if (dt.hour, dt.minute) == (0, 0) else f"{day}, {human_time(dt)}"


def humanize_dates(text: str) -> str:
    return re.sub(r"\b(\d{4}-\d{2}-\d{2})\b", lambda m: human_date(m.group(1)), text)


def words(token) -> str:
    return str(token or "").replace("_", " ").strip()


def cap(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def sentence(text) -> str:
    t = (text or "").strip()
    return t if not t or t[-1] in ".!?" else t + "."


_ABBR = re.compile(r"(?:\b(?:Dr|Mr|Mrs|Ms|St|vs|No)|\b[A-Z])\.$")


def sentences(text) -> list[str]:
    out: list[str] = []
    for part in re.split(r"(?<=[.!?])\s+", (text or "").strip()):
        if not part:
            continue
        if out and _ABBR.search(out[-1]):
            out[-1] += " " + part
        else:
            out.append(part)
    return out


def brief(parts, limit: int = 170) -> str:
    """Leading sentences of a text (or list of sentences) up to `limit` chars; always at least one."""
    sents = sentences(parts) if isinstance(parts, str) else list(parts)
    out = ""
    for s in sents:
        if out and len(out) + len(s) + 1 > limit:
            break
        out = f"{out} {s}".strip()
    return sentence(out)


MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]


def months_in(range_text: str) -> set[int]:
    idx = [MONTHS.index(t[:3].lower()) for t in re.findall(r"[A-Za-z]{3,}", range_text or "") if t[:3].lower() in MONTHS]
    if not idx:
        return set()
    a, b = idx[0], idx[-1]
    return {(a + i) % 12 for i in range((b - a) % 12 + 1)}


def render(template: str, params: list[str]) -> str:
    body = re.sub(r"\{\{(\d+)\}\}", lambda m: params[int(m.group(1)) - 1], template)
    body = re.sub(r"[ \t]{2,}", " ", body)
    body = re.sub(r"[ \t]+([.,!?])", r"\1", body)
    body = re.sub(r"\.\.+", ".", body)
    return body.strip()


# ---------------------------------------------------------------------------
# Context store + conversation state (in memory; single worker)
# ---------------------------------------------------------------------------

class Store:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.reset()

    def reset(self) -> None:
        with self.lock:
            self.contexts: dict[str, dict[str, dict]] = {s: {} for s in SCOPES}
            self.conversations: dict[str, dict] = {}
            self.merchant_state: dict[str, dict] = {}
            self.customer_state: dict[str, dict] = {}
            self.used_suppression: set[str] = set()

    def push(self, body) -> tuple[int, dict]:
        if not isinstance(body, dict):
            return 400, {"accepted": False, "reason": "malformed", "details": "body must be a JSON object"}
        scope = body.get("scope")
        if scope not in SCOPES:
            return 400, {"accepted": False, "reason": "invalid_scope", "details": f"scope must be one of {list(SCOPES)}"}
        cid, version, payload = body.get("context_id"), body.get("version"), body.get("payload")
        if not isinstance(cid, str) or not cid:
            return 400, {"accepted": False, "reason": "malformed", "details": "context_id must be a non-empty string"}
        if not isinstance(version, int) or isinstance(version, bool):
            return 400, {"accepted": False, "reason": "malformed", "details": "version must be an integer"}
        if not isinstance(payload, dict):
            return 400, {"accepted": False, "reason": "malformed", "details": "payload must be an object"}
        with self.lock:
            current = self.contexts[scope].get(cid)
            if current and current["version"] >= version:
                return 409, {"accepted": False, "reason": "stale_version", "current_version": current["version"]}
            self.contexts[scope][cid] = {"version": version, "payload": payload}
        return 200, {"accepted": True, "ack_id": f"ack_{cid}_v{version}", "stored_at": utcnow_iso()}

    def find(self, scope: str, cid) -> Optional[dict]:
        if not cid:
            return None
        rec = self.contexts[scope].get(cid)
        if rec:
            return rec["payload"]
        field = ID_FIELD[scope]
        for rec in self.contexts[scope].values():
            if rec["payload"].get(field) == cid:
                return rec["payload"]
        return None

    def counts(self) -> dict[str, int]:
        return {s: len(self.contexts[s]) for s in SCOPES}

    def mstate(self, merchant_id: str) -> dict:
        return self.merchant_state.setdefault(merchant_id, {
            "opted_out_until": None, "wait_until": None, "last_activity_at": None,
            "auto_reply_count": 0, "last_inbound": None, "inbound_repeat": 0, "unanswered": 0,
        })

    def cstate(self, customer_id: str) -> dict:
        return self.customer_state.setdefault(customer_id, {
            "last_activity_at": None, "opted_out": False,
            "auto_reply_count": 0, "last_inbound": None, "inbound_repeat": 0,
        })

    def sender_state(self, conv: dict, role: str) -> dict:
        """Per-sender counters (auto-replies, repeats) persist across conversations with the same party."""
        if role == "customer" and conv.get("customer_id"):
            return self.cstate(conv["customer_id"])
        if conv.get("merchant_id"):
            return self.mstate(conv["merchant_id"])
        return conv.setdefault("anon_sender", {"auto_reply_count": 0, "last_inbound": None, "inbound_repeat": 0})


store = Store()


# ---------------------------------------------------------------------------
# Context resolution
# ---------------------------------------------------------------------------

AUDIENCE = {"dentists": "patient", "pharmacies": "customer", "salons": "client", "gyms": "member", "restaurants": "customer"}
PLACE_NOUN = {"dentists": "dental clinic", "salons": "salon", "restaurants": "restaurant", "gyms": "gym", "pharmacies": "pharmacy"}
DEMAND_NOUN = {"dentists": "treatment", "salons": "service", "restaurants": "dish", "gyms": "class", "pharmacies": "product"}
RECALL_NOUN = {"dentists": "check-up", "gyms": "fitness check-in", "salons": "next appointment",
               "pharmacies": "health check", "restaurants": "next visit"}
WINBACK_ASK = {"restaurants": "Want us to reserve a table for you this week?",
               "pharmacies": "Want us to keep your usual items ready for pickup or delivery?"}
METRIC_LABEL = {"views": "profile views", "calls": "calls", "directions": "direction requests",
                "ctr": "click-through rate", "leads": "leads", "review_count": "Google reviews"}
THEME_LABEL = {"delivery_late": "late deliveries", "wait_time": "wait times", "saturday_wait": "Saturday waiting times"}
REMINDER_KINDS = {"recall_due", "appointment_tomorrow", "chronic_refill_due", "trial_followup"}
CUSTOMER_EVENT = {
    "recall_due": "is due for a recall visit", "appointment_tomorrow": "has an appointment tomorrow",
    "chronic_refill_due": "is due for a refill", "customer_lapsed_soft": "hasn't visited in a while",
    "customer_lapsed_hard": "hasn't visited in a while", "trial_followup": "tried a session and hasn't booked yet",
    "wedding_package_followup": "is due for a pre-wedding follow-up",
}


def digest_item(category: dict, trigger: dict) -> Optional[dict]:
    payload = trigger.get("payload") or {}
    ref = payload.get("top_item_id") or payload.get("digest_item_id") or payload.get("alert_id")
    if not ref:
        return None
    return next((it for it in category.get("digest") or [] if it.get("id") == ref), None)


def consent_ok(customer: dict, kind: str) -> bool:
    consent = customer.get("consent") or {}
    if not consent.get("opted_in_at") or not consent.get("scope"):
        return False
    if kind in REMINDER_KINDS and (customer.get("preferences") or {}).get("reminder_opt_in") is False:
        return False
    return True


class Ctx:
    """Derived view over the 4 contexts, shared by the template builders."""

    def __init__(self, category: dict, merchant: dict, trigger: dict, customer: Optional[dict] = None):
        self.category, self.merchant, self.trigger, self.customer = category or {}, merchant or {}, trigger or {}, customer
        self.slug = self.category.get("slug") or self.merchant.get("category_slug", "")
        self.kind = self.trigger.get("kind", "")
        self.payload = self.trigger.get("payload") or {}
        self.placeholder = bool(self.payload.get("placeholder"))
        self.ident = self.merchant.get("identity") or {}
        self.perf = self.merchant.get("performance") or {}
        self.peer = self.category.get("peer_stats") or {}
        self.agg = self.merchant.get("customer_aggregate") or {}
        self.sub = self.merchant.get("subscription") or {}
        self.offers = [o["title"] for o in self.merchant.get("offers") or [] if o.get("status") == "active" and o.get("title")]
        self.item = digest_item(self.category, self.trigger)
        self.audience = AUDIENCE.get(self.slug, "customer")

    @property
    def name(self) -> str:
        return self.ident.get("name") or "your business"

    @property
    def greet(self) -> str:
        first = re.sub(r"^Dr\.?\s*", "", (self.ident.get("owner_first_name") or "").strip())
        if not first:
            return f"Hi {self.name} team"
        return f"Dr. {first}" if self.slug == "dentists" else f"Hi {first}"

    def offer_hook(self) -> str:
        if self.offers:
            return f"your {self.offers[0]} offer"
        for o in self.category.get("offer_catalog") or []:
            if o.get("type") == "service_at_price":
                return f"a service-price offer like {o['title']}"
        return "one simple service + price offer"

    def perf_counts(self) -> str:
        return f"{num(self.perf.get('views', 0))} profile views and {num(self.perf.get('calls', 0))} calls"

    def peer_counts(self) -> str:
        return f"peer average: {num(self.peer.get('avg_views_30d', 0))} views, {num(self.peer.get('avg_calls_30d', 0))} calls"

    def perf_fact(self) -> str:
        return f"Your listing got {self.perf_counts()} in the last 30 days."

    def metric_vs_peer(self, metric: str) -> str:
        mine = self.perf.get(metric)
        peer = self.peer.get("avg_ctr") if metric == "ctr" else self.peer.get(f"avg_{metric}_30d")
        if mine is None or peer is None:
            return ""
        if metric == "ctr":
            return f"Your click-through rate is {mine * 100:.1f}% vs a peer average of {peer * 100:.1f}%."
        return f"Your 30-day total is {num(mine)} {METRIC_LABEL.get(metric, words(metric))} vs a peer average of {num(peer)}."

    def beat_for(self, months: set[int]) -> Optional[dict]:
        if not months:
            return None
        return next((b for b in self.category.get("seasonal_beats") or [] if months_in(b.get("month_range", "")) & months), None)

    def segment_line(self, item: dict) -> str:
        seg = item.get("patient_segment") or ""
        n = self.agg.get(seg[:-1] + "_count") if seg.endswith("s") else None
        if not n:
            return ""
        return f"Directly relevant to the {num(n)} {words(seg).replace('high risk', 'high-risk')} on your roster."

    # customer-facing helpers
    def cust_names(self) -> tuple[str, str]:
        """(greeting, possessive for the person the service is for)."""
        ident = (self.customer or {}).get("identity") or {}
        name = (ident.get("name") or "").strip()
        channel = ((self.customer or {}).get("preferences") or {}).get("channel", "")
        lang = (ident.get("language_pref") or "").lower()
        m = re.match(r"^(.*?)\s*\(parent:\s*(.*?)\)\s*$", name)
        if m:
            return f"Hi {m.group(2).strip()}", f"{m.group(1).strip()}'s"
        if not name or name.startswith("("):
            return "Namaste", "your"
        if "via_son" in channel or "via_daughter" in channel or ident.get("senior_citizen"):
            base = re.sub(r"^(Mr|Mrs|Ms)\.?\s+", "", name)
            return "Namaste", f"{base} ji's"
        return f"{'Namaste' if lang == 'hi' else 'Hi'} {name}", "your"

    def cust_pref(self, key: str, default=None):
        return ((self.customer or {}).get("preferences") or {}).get(key, default)

    def cust_rel(self) -> dict:
        return (self.customer or {}).get("relationship") or {}

    def slot_pref(self) -> str:
        raw = words(self.cust_pref("preferred_slots") or "")
        if not raw:
            return "a time that suits you"
        raw = re.sub(r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", lambda m: m.group(1).title(), raw)
        return raw + "s" if raw.endswith(("morning", "evening", "afternoon")) else raw


# ---------------------------------------------------------------------------
# Fallback templates. Each builder returns (template_name, params, cta, rationale).
# Used when the LLM path is unavailable; body is always render(TEMPLATES[name], params).
# ---------------------------------------------------------------------------

TEMPLATES = {
    "vera_research_digest_v1": "{{1}}, one research update worth 2 minutes: {{2}} ({{3}}). {{4}} Want me to draft a short {{5}} WhatsApp on this that you can forward? Reply YES.",
    "vera_compliance_alert_v1": "{{1}}, compliance heads-up: {{2}} ({{3}}). {{4}} Want a 3-point checklist to be ready before {{5}}? Reply YES.",
    "vera_cde_invite_v1": "{{1}}, a CDE session worth blocking: {{2}}, {{3}} ({{4}}). {{5}} Want me to add it to your calendar with the details? Reply YES.",
    "vera_supply_alert_v1": "{{1}}, urgent: {{2}} ({{3}}). {{4}} Want me to draft the customer WhatsApp and a replacement-pickup note? Reply YES.",
    "vera_seasonal_shift_v1": "{{1}}, seasonal shift for {{2}}: {{3}}. {{4}} Want me to draft a shelf and WhatsApp plan for this week? Reply YES.",
    "vera_festival_prep_v1": "{{1}}, {{2}} is {{3}} away ({{4}}). {{5}} Want me to draft a {{2}} post built around {{6}}? Reply YES.",
    "vera_local_event_v1": "{{1}}, {{2}} at {{3}} today, {{4}}. {{5}} Want me to draft {{6}}? Reply YES.",
    "vera_competitor_alert_v1": "{{1}}, heads-up: {{2}}. {{3}} Want me to sharpen your Google listing so searchers compare you favourably? Reply YES.",
    "vera_perf_dip_v1": "{{1}}, your {{2}} dropped {{3}} {{4}}. {{5}} Want me to check what changed and suggest 2 quick fixes? Reply YES.",
    "vera_perf_spike_v1": "{{1}}, your {{2}} {{3}} up {{4}} {{5}}. {{6}} Want me to line up a follow-up post while it's working? Reply YES.",
    "vera_seasonal_dip_v1": "{{1}}, your {{2}} {{3}} down {{4}} this week, and that's expected: {{5}}. Want me to draft a retention push for your {{6}} while new demand is slow? Reply YES.",
    "vera_milestone_v1": "{{1}}, you're at {{2}} {{3}}, just {{4}} short of {{5}}. Want me to draft a thank-you post that invites happy customers to leave a review? Reply YES.",
    "vera_review_theme_v1": "{{1}}, {{2}} reviews in the last 30 days mention {{3}}{{4}}. {{5}} Want me to draft a polite public reply plus one fix you can announce? Reply YES.",
    "vera_checkin_v1": "{{1}}, it's been {{2}} days since we last spoke{{3}}. {{4}} Want a 2-minute snapshot of what's working and what isn't? Reply YES.",
    "vera_curious_ask_v1": "{{1}}, quick question: {{2}} Reply in one line and I'll turn it into a Google post for {{3}}.",
    "vera_renewal_v1": "{{1}}, your {{2}} plan renews in {{3}} days{{4}}. Renewing keeps your Google posts, offers and review replies running. Want me to keep everything going without a break? Reply YES.",
    "vera_winback_v1": "{{1}}, it's been {{2}} days since your plan lapsed{{3}}. {{4}} Want me to restart your profile updates this week? Reply YES.",
    "vera_gbp_verify_v1": "{{1}}, {{2}} is still unverified on Google. {{3}} Verification is by {{4}}. Want me to walk you through it? Reply YES.",
    "vera_planning_followup_v1": "{{1}}, on your {{2}} idea: I'm drafting a first version built around {{3}}. Reply YES and I'll share it here for your edits.",
    "vera_account_update_v1": "{{1}}, quick update on {{2}}: {{3}} in the last 30 days ({{4}}). {{5}} Want me to {{6}}? Reply YES.",
    "merchant_recall_slots_v1": "{{1}}, {{2}} here. {{3}} is due{{4}}. Slots open: {{5}}. {{6}} Reply with the slot number to book, or tell us a time that suits you.",
    "merchant_recall_v1": "{{1}}, {{2}} here. {{3}} is due{{4}}. {{5}} Reply YES and we'll share open slots for {{6}}.",
    "merchant_appointment_reminder_v1": "{{1}}, {{2}} here, a reminder about {{3}} appointment tomorrow{{4}}. Reply YES to confirm, or tell us if you need a different time.",
    "merchant_refill_reminder_v1": "{{1}}, {{2}} here. {{3}} ({{4}}) will run out around {{5}}. {{6}} Reply CONFIRM and we'll keep the same pack ready{{7}}.",
    "merchant_winback_v1": "{{1}}, {{2}} here. It's been {{3}} since {{4}} last visit, and that's completely fine. {{5}} {{6}} Reply YES.",
    "merchant_trial_followup_v1": "{{1}}, {{2}} here. Thanks for coming in for {{3}} on {{4}}. The next session is {{5}}. Want us to save a spot? Reply YES.",
    "merchant_bridal_followup_v1": "{{1}}, {{2}} here. {{3}} days to your wedding on {{4}}, so this is a good window to start {{5}}. Want us to book your first session on a {{6}}? Reply YES.",
    "merchant_customer_update_v1": "{{1}}, {{2}} here. {{3}} Reply YES and we'll take care of it.",
    # LLM-composed: opening (salutation + why now), detail (anchors + judgment), ask (single CTA)
    "vera_compose_v1": "{{1}} {{2}} {{3}}",
    "merchant_compose_v1": "{{1}} {{2}} {{3}}",
}


def _b_account(c: Ctx):
    """Generic merchant update anchored on the merchant's own numbers; used for placeholders and unknown kinds."""
    k = c.kind
    framing, action = "", "share 2 quick ideas to lift calls this week"
    if k == "festival_upcoming":
        framing, action = "Festival season is coming up.", f"draft a festival post built around {c.offer_hook()}"
    elif k == "milestone_reached":
        framing, action = "Your listing just hit a new milestone.", "draft a thank-you post that invites happy customers to leave a review"
    elif k == "dormant_with_vera":
        framing, action = "It's been a while since we last spoke.", "send a 2-minute snapshot of what's working"
    elif k in ("perf_dip", "perf_spike", "seasonal_perf_dip"):
        action = "check what's driving your calls and suggest 2 quick fixes"
    elif k == "review_theme_emerged":
        action = "draft replies to your latest reviews"
    elif k in CUSTOMER_EVENT:
        framing, action = f"One of your {c.audience}s {CUSTOMER_EVENT[k]}.", "draft the message for you to send"
    elif (c.trigger.get("source") == "external") and k:
        framing = f"Heads-up: {words(k)} in {c.ident.get('city', 'your city')}."
    params = [c.greet, c.name, c.perf_counts(), c.peer_counts(), framing, action]
    return ("vera_account_update_v1", params, "binary_yes_no",
            f"{k or 'update'} trigger with little payload detail: anchored on the merchant's real 30-day numbers vs peers "
            "instead of inventing specifics; single YES CTA.")


def _b_research(c: Ctx):
    it = c.item
    if not it:
        return _b_account(c)
    sents = sentences(it.get("summary"))
    if sents and it.get("trial_n"):
        sents[0] = sents[0].rstrip(".") + f" (n={num(it['trial_n'])})."
    detail = f"{brief(sents)} {c.segment_line(it)}".strip()
    params = [c.greet, it.get("title", ""), it.get("source", "this week's digest"), detail, c.audience]
    return ("vera_research_digest_v1", params, "binary_yes_no",
            f"{c.kind}: cites digest item {it.get('id')} with its source and numbers"
            f"{', tied to the merchant cohort' if c.segment_line(it) else ''}; offers to draft {c.audience} content "
            "(reciprocity); single YES CTA.")


def _b_compliance(c: Ctx):
    it = c.item
    if not it:
        return _b_account(c)
    deadline = c.payload.get("deadline_iso") or it.get("date")
    detail = " ".join(x for x in (brief(it.get("summary"), 140), sentence(it.get("actionable"))) if x)
    params = [c.greet, it.get("title", ""), it.get("source", "regulator circular"), detail,
              human_date(deadline) if deadline else "the deadline"]
    return ("vera_compliance_alert_v1", params, "binary_yes_no",
            f"{c.kind}: compliance item {it.get('id')} with source citation and deadline (loss aversion); "
            "offers a ready checklist; single YES CTA.")


def _b_cde(c: Ctx):
    it = c.item
    if not it:
        return _b_account(c)
    when = human_datetime(it["date"]) if it.get("date") else "date to be announced"
    credits = c.payload.get("credits") or it.get("credits")
    fee = it.get("actionable") or words(c.payload.get("fee"))
    meta = "; ".join(x for x in (f"{credits} CDE credits" if credits else "", fee) if x) or it.get("source", "")
    params = [c.greet, it.get("title", ""), when, meta, brief(it.get("summary"))]
    return ("vera_cde_invite_v1", params, "binary_yes_no",
            f"{c.kind}: CDE event {it.get('id')} with date, credits and fee from the digest; low-effort calendar offer; single YES CTA.")


def _b_supply(c: Ctx):
    it, p = c.item or {}, c.payload
    if p.get("affected_batches") and p.get("molecule"):
        what = f"voluntary recall on {p['molecule']} batches {', '.join(p['affected_batches'])}"
        if p.get("manufacturer"):
            what += f" by {p['manufacturer']}"
    elif it:
        what = it.get("title", "")
    else:
        return _b_account(c)
    risk = next((s for s in sentences(it.get("summary")) if "risk" in s.lower()), "")
    n = c.agg.get("chronic_rx_count")
    count = f"You have {num(n)} chronic-Rx customers on file to check." if n else ""
    detail = " ".join(x for x in (sentence(risk), sentence(it.get("actionable")), count) if x)
    params = [c.greet, what, it.get("source", "supplier alert"), detail]
    return ("vera_supply_alert_v1", params, "binary_yes_no",
            f"{c.kind}: batch-level recall facts from the trigger payload plus the digest source; bounded-risk framing; "
            "offers the complete customer workflow; single YES CTA.")


def _b_category_seasonal(c: Ctx):
    parsed = []
    for t in c.payload.get("trends") or []:
        m = re.match(r"^(.*?)_demand_([+-]?\d+)$", str(t))
        if m:
            delta = m.group(2) if m.group(2)[0] in "+-" else "+" + m.group(2)
            parsed.append(f"{words(m.group(1))} {delta}%")
    if not parsed:
        return _b_account(c)
    tip = next((sentence(i.get("actionable")) for i in c.category.get("digest") or []
                if i.get("kind") == "seasonal" and i.get("actionable")), "Worth adjusting shelf placement this week.")
    params = [c.greet, words(c.payload.get("season")) or "this season", ", ".join(parsed), tip]
    return ("vera_seasonal_shift_v1", params, "binary_yes_no",
            f"{c.kind}: demand shifts quoted from the trigger payload, paired with the category's seasonal action; single YES CTA.")


def _b_festival(c: Ctx):
    fest, date, days = c.payload.get("festival"), c.payload.get("date"), c.payload.get("days_until")
    if not fest:
        return _b_account(c)
    dt = parse_ts(date) if date and "T" in date else (datetime.strptime(date[:10], "%Y-%m-%d") if date else None)
    beat = c.beat_for({dt.month - 1}) if dt else None
    beat_line = f"Seasonal pattern for {beat['month_range']}: {beat['note']}." if beat else ""
    params = [c.greet, fest, f"{days} days" if days is not None else "a few weeks", human_date(date) or "date TBC",
              beat_line, c.offer_hook()]
    return ("vera_festival_prep_v1", params, "binary_yes_no",
            f"{c.kind}: festival date and countdown from the payload, category seasonal beat for that month, "
            "built on the merchant's own offer; single YES CTA.")


def _b_local_event(c: Ctx):
    p = c.payload
    if not p.get("match"):
        return _b_account(c)
    t = parse_ts(p.get("match_time_iso"))
    weeknight = bool(p.get("is_weeknight"))
    insight = ""
    for it in c.category.get("digest") or []:
        if "ipl" in f"{it.get('title', '')} {it.get('summary', '')}".lower():
            sents = sentences(it.get("summary"))
            pick = next((s for s in sents if ("weeknight" in s.lower()) == weeknight), sents[0] if sents else "")
            insight = f"{pick.rstrip('.')} ({it.get('source', 'category data')})." if pick else ""
            break
    hook = c.offer_hook()
    weekday_only = bool(c.offers) and re.search(r"\b(Mon|Tue|Wed|Thu|Fri)\w*\s*-\s*(Mon|Tue|Wed|Thu|Fri)\w*\b|weekday",
                                                 c.offers[0], re.I)
    if weeknight:
        action = f"a match-night post around {hook}"
    elif weekday_only:
        action = "a delivery-first weekend special instead of a dine-in match promo"
    else:
        action = f"a delivery-first push around {hook} instead of a dine-in match promo"
    params = [c.greet, p["match"], p.get("venue") or c.ident.get("city", "your city"),
              human_time(t) if t else "tonight", insight, action]
    return ("vera_local_event_v1", params, "binary_yes_no",
            f"{c.kind}: match details from the payload plus the category's IPL data point for "
            f"{'weeknight' if weeknight else 'weekend'} matches (judgment, not just a promo); single YES CTA.")


def _b_competitor(c: Ctx):
    p = c.payload
    if p.get("competitor_name"):
        what = f"{p['competitor_name']} opened " + (f"{p['distance_km']} km from you" if p.get("distance_km") else "nearby")
        if p.get("opened_date"):
            what += f" on {human_date(p['opened_date'])}"
        if p.get("their_offer"):
            what += f", advertising {p['their_offer']}"
    else:
        what = f"a new {PLACE_NOUN.get(c.slug, 'business')} listing has opened near you in {c.ident.get('locality', 'your area')}"
    stance = (f"Your {c.offers[0]} is live, so make sure searchers see it first." if c.offers
              else "You have no active offer right now, so searchers comparing the two only see theirs.")
    params = [c.greet, what, stance]
    return ("vera_competitor_alert_v1", params, "binary_yes_no",
            f"{c.kind}: competitor facts only from the payload (none invented), contrasted with the merchant's "
            "own offer status (loss aversion); single YES CTA.")


def _b_perf(c: Ctx, sign: int):
    p = c.payload
    metric, delta = p.get("metric"), p.get("delta_pct")
    baseline, driver = p.get("vs_baseline"), p.get("likely_driver")
    if metric is None or delta is None:
        d7 = c.perf.get("delta_7d") or {}
        moves = [(k[:-4], v) for k, v in d7.items() if k.endswith("_pct") and isinstance(v, (int, float)) and v * sign > 0]
        if not moves:
            return _b_account(c)
        metric, delta = max(moves, key=lambda kv: abs(kv[1]))
        window, baseline, driver = "this week vs last week", None, None
    else:
        window = "over the last " + re.sub(r"^(\d+)d$", r"\1 days", str(p.get("window", "7d")))
    label = METRIC_LABEL.get(metric, words(metric))
    if sign < 0:
        params = [c.greet, label, pct(delta), window + (f" (usual level: about {baseline})" if baseline else ""),
                  c.metric_vs_peer(metric)]
        return ("vera_perf_dip_v1", params, "binary_yes_no",
                f"{c.kind}: exact {metric} drop and window, compared with the peer benchmark (loss aversion); "
                "offers a diagnosis; single YES CTA.")
    line = f"Likely driver: your {words(driver)}." if driver else c.metric_vs_peer(metric)
    verb = "is" if metric == "ctr" else "are"
    params = [c.greet, label, verb, pct(delta), window, line]
    return ("vera_perf_spike_v1", params, "binary_yes_no",
            f"{c.kind}: exact {metric} rise{' and its likely driver' if driver else ''}; proposes capitalising "
            "with a post while momentum lasts; single YES CTA.")


def _b_seasonal_dip(c: Ctx):
    p = c.payload
    if p.get("delta_pct") is None:
        return _b_account(c)
    metric = p.get("metric", "views")
    note_months = {MONTHS.index(t) for t in str(p.get("season_note", "")).lower().split("_") if t in MONTHS}
    beat = c.beat_for(note_months)
    reason = f"{beat['month_range']} is typically the {beat['note']}" if beat else words(p.get("season_note")) or "a seasonal pattern"
    members = c.agg.get("total_active_members")
    base = f"{num(members)} active members" if members else f"regular {c.audience}s"
    params = [c.greet, METRIC_LABEL.get(metric, words(metric)), "is" if metric == "ctr" else "are",
              pct(p["delta_pct"]), reason, base]
    return ("vera_seasonal_dip_v1", params, "binary_yes_no",
            f"{c.kind}: reframes the {metric} dip with the category's seasonal pattern (anxiety pre-emption) "
            "and redirects to retention; single YES CTA.")


def _b_milestone(c: Ctx):
    p = c.payload
    metric, now_v, target = p.get("metric"), p.get("value_now"), p.get("milestone_value")
    if metric is None or now_v is None or target is None or target <= now_v:
        return _b_account(c)
    params = [c.greet, num(now_v), METRIC_LABEL.get(metric, words(metric)), num(target - now_v), num(target)]
    return ("vera_milestone_v1", params, "binary_yes_no",
            f"{c.kind}: exact count and gap to the milestone (goal proximity); offers a review-invite post; single YES CTA.")


def _b_review_theme(c: Ctx):
    p = dict(c.payload)
    if not p.get("theme"):
        neg = sorted((t for t in c.merchant.get("review_themes") or [] if t.get("sentiment") == "neg"),
                     key=lambda t: -(t.get("occurrences_30d") or 0))
        if not neg:
            return _b_account(c)
        p = neg[0]
    n = p.get("occurrences_30d")
    params = [c.greet, num(n) if n else "Several", THEME_LABEL.get(p["theme"], words(p["theme"])),
              " and the count is rising" if p.get("trend") == "rising" else "",
              f'One reviewer wrote: "{p["common_quote"]}".' if p.get("common_quote") else ""]
    return ("vera_review_theme_v1", params, "binary_yes_no",
            f"{c.kind}: review count, theme and a real quote from the data; offers reply + fix; single YES CTA.")


def _b_dormant(c: Ctx):
    days = c.payload.get("days_since_last_merchant_message")
    if days is None:
        return _b_account(c)
    topic = c.payload.get("last_topic")
    params = [c.greet, num(days), f" (last time it was about {words(topic)})" if topic else "", c.perf_fact()]
    return ("vera_checkin_v1", params, "binary_yes_no",
            f"{c.kind}: re-opens after {days} quiet days with the merchant's real numbers (reciprocity); single YES CTA.")


def _b_curious(c: Ctx):
    noun = DEMAND_NOUN.get(c.slug, "service")
    ask = f"which {noun} got the most requests this week?"
    if c.offers:
        ask += f" Is it still the {c.offers[0]}?"
    params = [c.greet, ask, c.name]
    return ("vera_curious_ask_v1", params, "open_ended",
            f"{c.kind}: asking-the-merchant lever with a concrete guess from their active offer; effort externalised "
            "(we write the post); open-ended reply.")


def _b_renewal(c: Ctx):
    p = c.payload
    days = p.get("days_remaining", c.sub.get("days_remaining"))
    if days is None:
        return _b_account(c)
    amount = p.get("renewal_amount")
    params = [c.greet, p.get("plan") or c.sub.get("plan") or "current", num(days), f" ({inr(amount)})" if amount else ""]
    return ("vera_renewal_v1", params, "binary_yes_no",
            f"{c.kind}: exact days left and renewal amount from the payload; continuity framing; single YES CTA.")


def _b_winback(c: Ctx):
    p = c.payload
    days = p.get("days_since_expiry", c.sub.get("days_since_expiry"))
    if days is None:
        return _b_account(c)
    dip, lapsed = p.get("perf_dip_pct"), p.get("lapsed_customers_added_since_expiry")
    params = [c.greet, num(days), f", and your numbers are down {pct(dip)} since" if dip else "",
              f"{num(lapsed)} of your customers have lapsed in that time." if lapsed else c.perf_fact()]
    return ("vera_winback_v1", params, "binary_yes_no",
            f"{c.kind}: days since expiry, performance drop and lapsed-customer count from the payload (loss aversion); "
            "single YES CTA.")


def _b_gbp(c: Ctx):
    p = c.payload
    uplift = p.get("estimated_uplift_pct")
    params = [c.greet, c.name,
              f"Verifying could lift your listing's results by an estimated {pct(uplift)}." if uplift
              else "Google reviews every edit on unverified listings, which slows your updates.",
              words(p.get("verification_path")) or "postcard or phone call"]
    return ("vera_gbp_verify_v1", params, "binary_yes_no",
            f"{c.kind}: verification status, estimated uplift and verification path from the payload; "
            "offers a guided walkthrough; single YES CTA.")


def _b_planning(c: Ctx):
    topic = c.payload.get("intent_topic")
    if not topic:
        return _b_account(c)
    base = f"your {c.offers[0]}" if c.offers else f"your current {DEMAND_NOUN.get(c.slug, 'service')}s"
    params = [c.greet, re.sub(r"\s+package$", "", words(topic)), base]
    return ("vera_planning_followup_v1", params, "binary_yes_no",
            f"{c.kind}: merchant already said yes to the idea, so this moves straight to action (drafting) "
            "instead of re-qualifying; single YES CTA.")


# customer-facing builders ---------------------------------------------------

def _b_customer_generic(c: Ctx):
    greet, poss = c.cust_names()
    k = c.kind
    if k == "chronic_refill_due":
        framing = "Your regular refill is due soon." if c.slug == "pharmacies" else "It's almost time for your next visit."
    elif k in ("customer_lapsed_soft", "customer_lapsed_hard"):
        framing = "It's been a while since your last visit."
    elif k == "trial_followup":
        framing = "Thanks for trying us out."
    elif k == "recall_due":
        framing = f"{cap(poss)} {RECALL_NOUN.get(c.slug, 'next visit')} is due."
    else:
        framing = "We have an update for you."
    last = c.cust_rel().get("last_visit")
    extra = f"Your last visit was on {human_date(last)}." if last and poss == "your" else ""
    offer = f"Current offer: {c.offers[0]}." if c.offers else ""
    params = [greet, c.name, " ".join(x for x in (framing, extra, offer) if x)]
    return ("merchant_customer_update_v1", params, "binary_yes_no",
            f"{k}: customer-facing note on the merchant's behalf using only relationship data on file; single YES CTA.")


def _b_recall(c: Ctx):
    greet, poss = c.cust_names()
    p = c.payload
    service = words(p.get("service_due")) or RECALL_NOUN.get(c.slug, "check-up")
    service = re.sub(r"(\d+) month", r"\1-month", service)
    last = p.get("last_service_date") or c.cust_rel().get("last_visit")
    since = f" (last visit: {human_date(last)})" if last else ""
    offer = f"Current offer: {c.offers[0]}." if c.offers else ""
    slots = [s.get("label") for s in p.get("available_slots") or [] if s.get("label")][:3]
    if slots:
        slot_text = " or ".join(f"{i}) {s}" for i, s in enumerate(slots, 1))
        params = [greet, c.name, f"{cap(poss)} {service}", since, slot_text, offer]
        return ("merchant_recall_slots_v1", params, "multi_choice_slot",
                f"{c.kind}: real recall service, last visit and open slots from the payload, merchant's live offer; "
                "numbered slot choice suits a booking flow.")
    params = [greet, c.name, f"{cap(poss)} {service}", since, offer, c.slot_pref()]
    return ("merchant_recall_v1", params, "binary_yes_no",
            f"{c.kind}: recall due with last-visit date; offers slots in the customer's preferred window; single YES CTA.")


def _b_appointment(c: Ctx):
    greet, poss = c.cust_names()
    p = c.payload
    label = p.get("slot_label") or p.get("time_label") or p.get("label")
    t = parse_ts(p.get("appointment_iso") or p.get("slot_iso") or p.get("iso"))
    at = f" at {label}" if label else (f" at {human_time(t)}" if t else "")
    params = [greet, c.name, poss, at]
    return ("merchant_appointment_reminder_v1", params, "binary_yes_no",
            f"{c.kind}: appointment reminder on the merchant's behalf; confirm-or-reschedule CTA.")


def _b_refill(c: Ctx):
    greet, poss = c.cust_names()
    p = c.payload
    mols = p.get("molecule_list") or []
    if not mols:
        return _b_customer_generic(c)
    senior = ((c.customer or {}).get("identity") or {}).get("senior_citizen")
    relevant = [o for o in c.offers if "deliver" in o.lower() or ("senior" in o.lower() and senior)]
    params = [greet, c.name, f"{cap(poss)} medicines", ", ".join(mols), human_date(p.get("stock_runs_out_iso")) or "soon",
              f"Active offers: {', '.join(relevant)}." if relevant else "",
              " for home delivery to your saved address" if p.get("delivery_address_saved") else ""]
    return ("merchant_refill_reminder_v1", params, "binary_confirm_cancel",
            f"{c.kind}: exact molecules and run-out date from the payload, only offers the customer qualifies for; "
            "CONFIRM CTA to dispatch.")


def _b_lapsed(c: Ctx):
    greet, poss = c.cust_names()
    p = c.payload
    days = p.get("days_since_last_visit")
    gap = f"about {max(1, round(days / 7))} weeks" if days else "a while"
    focus = words(p.get("previous_focus"))
    offer = c.offers[0] if c.offers else ""
    if offer and focus:
        hook = f"We're running {offer} right now, an easy way back into your {focus} routine."
    elif offer:
        hook = f"We're running {offer} right now."
    elif focus:
        hook = f"We'd love to help you pick your {focus} routine back up."
    else:
        hook = ""
    ask = WINBACK_ASK.get(c.slug, "Want us to hold a slot for you this week?")
    params = [greet, c.name, gap, poss, hook, ask]
    return ("merchant_winback_v1", params, "binary_yes_no",
            f"{c.kind}: no-shame winback with the real gap{', past goal' if focus else ''}"
            f"{' and live offer' if offer else ''}; single YES CTA.")


def _b_trial(c: Ctx):
    greet, poss = c.cust_names()
    p = c.payload
    opts = [o.get("label") for o in p.get("next_session_options") or [] if o.get("label")]
    if not opts:
        return _b_customer_generic(c)
    services = c.cust_rel().get("services_received") or []
    service = words(services[-1]) if services else "trial"
    params = [greet, c.name, f"{poss} {service}", human_date(p.get("trial_date")) or "your first visit", " or ".join(opts[:2])]
    return ("merchant_trial_followup_v1", params, "binary_yes_no",
            f"{c.kind}: trial date and the real next session from the payload; single YES CTA.")


def _b_bridal(c: Ctx):
    greet, _ = c.cust_names()
    p = c.payload
    days, wedding = p.get("days_to_wedding"), p.get("wedding_date") or c.cust_pref("wedding_date")
    if days is None or not wedding:
        return _b_customer_generic(c)
    raw = str(p.get("next_step_window_open") or "")
    m = re.search(r"(\d+)day", raw)
    step = words(re.sub(r"_?\d+day", "", raw)) or "your pre-wedding prep"
    step = f"the {m.group(1)}-day {step}" if m else step
    day = c.slot_pref().rstrip("s") if c.cust_pref("preferred_slots") else "day that suits you"
    params = [greet, c.name, num(days), human_date(wedding), step, day]
    return ("merchant_bridal_followup_v1", params, "binary_yes_no",
            f"{c.kind}: wedding countdown and next-step window from the payload, customer's preferred day; single YES CTA.")


MERCHANT_BUILDERS = {
    "research_digest": _b_research, "research_digest_release": _b_research,
    "category_research_digest_release": _b_research, "category_trend_movement": _b_research,
    "regulation_change": _b_compliance, "cde_opportunity": _b_cde, "supply_alert": _b_supply,
    "category_seasonal": _b_category_seasonal, "festival_upcoming": _b_festival, "ipl_match_today": _b_local_event,
    "competitor_opened": _b_competitor, "perf_dip": lambda c: _b_perf(c, -1), "perf_spike": lambda c: _b_perf(c, 1),
    "seasonal_perf_dip": _b_seasonal_dip, "milestone_reached": _b_milestone, "review_theme_emerged": _b_review_theme,
    "dormant_with_vera": _b_dormant, "curious_ask_due": _b_curious, "renewal_due": _b_renewal,
    "winback_eligible": _b_winback, "gbp_unverified": _b_gbp, "active_planning_intent": _b_planning,
}
CUSTOMER_BUILDERS = {
    "recall_due": _b_recall, "appointment_tomorrow": _b_appointment, "chronic_refill_due": _b_refill,
    "customer_lapsed_soft": _b_lapsed, "customer_lapsed_hard": _b_lapsed, "trial_followup": _b_trial,
    "wedding_package_followup": _b_bridal, "bridal_followup": _b_bridal,
}
ITEM_KIND_BUILDERS = {"compliance": _b_compliance, "cde": _b_cde, "alert": _b_supply}


def _pick_builder(c: Ctx, customer_facing: bool):
    if customer_facing:
        return CUSTOMER_BUILDERS.get(c.kind, _b_customer_generic)
    if c.kind in MERCHANT_BUILDERS:
        return MERCHANT_BUILDERS[c.kind]
    if c.item:
        return ITEM_KIND_BUILDERS.get(c.item.get("kind"), _b_research)
    return _b_account


def compose_template(category: dict, merchant: dict, trigger: dict, customer: Optional[dict] = None) -> dict:
    """Deterministic, fact-checked composition from templates (fallback and LLM reference draft)."""
    c = Ctx(category, merchant, trigger, customer)
    customer_facing = trigger.get("scope") == "customer" and customer is not None
    name, params, cta, rationale = _pick_builder(c, customer_facing)(c)
    params = [humanize_dates(str(p)) for p in params]
    return {
        "body": render(TEMPLATES[name], params),
        "cta": cta,
        "send_as": "merchant_on_behalf" if customer_facing else "vera",
        "suppression_key": trigger.get("suppression_key") or f"{c.kind}:{merchant.get('merchant_id')}:{trigger.get('id')}",
        "rationale": rationale,
        "template_name": name,
        "template_params": params,
        "composer": "template",
    }


def compose(category: dict, merchant: dict, trigger: dict, customer: Optional[dict] = None,
            deadline: Optional[float] = None) -> dict:
    """Brief §7.1 entry point: LLM composition validated against the facts, template fallback otherwise.

    Returns body, cta, send_as, suppression_key, rationale (+ template_name/template_params, composer).
    `deadline` is a time.monotonic() value; LLM work that can't finish before it is skipped.
    """
    base = compose_template(category, merchant, trigger, customer)
    try:
        llm_msg = _compose_llm(Ctx(category, merchant, trigger, customer), base, deadline)
    except Exception as e:  # never let the LLM path break a send
        base["llm_problems"] = [f"composer error: {type(e).__name__}"]
        return base
    if llm_msg.get("composer") == "llm":
        return llm_msg
    base["llm_problems"] = llm_msg.get("problems", [])
    return base


# ---------------------------------------------------------------------------
# LLM client: disk-cached (which is also what makes compose() deterministic), cost-logged, budget-capped
# ---------------------------------------------------------------------------

CACHE_ROOT = Path(__file__).with_name(".cache")
LLM_CACHE_DIR = CACHE_ROOT / "llm"
LEDGER = CACHE_ROOT / "llm_ledger.jsonl"
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-5.6-luna")
LLM_TIMEOUT_S = float(os.environ.get("VERA_LLM_TIMEOUT_S", "8"))
PRICE_IN = float(os.environ.get("LLM_PRICE_IN", "0.20"))    # $ per 1M input tokens
PRICE_OUT = float(os.environ.get("LLM_PRICE_OUT", "1.20"))  # $ per 1M output tokens
SPEND_CAP = float(os.environ.get("VERA_SPEND_CAP_USD", "4.50"))
# "disk" (dev): responses persist in .cache/llm. "memory" (production): nothing touches disk and /v1/teardown wipes it,
# as the brief forbids persisting context data after the test. submission_cache/ is a read-only snapshot of the
# responses behind submission.jsonl, so compose() reproduces it exactly in either mode.
CACHE_MODE = os.environ.get("VERA_CACHE_MODE", "disk")
SHIPPED_CACHE_DIR = Path(__file__).with_name("submission_cache")
_mem_cache: dict[str, dict] = {}
CACHE_KEYS_USED: set[str] = set()  # read by tools/export_submission_cache.py
_llm_lock = threading.Lock()
_llm_client = None
_spent: Optional[float] = None


def _cache_get(key: str) -> Optional[dict]:
    paths = [SHIPPED_CACHE_DIR / f"{key}.json"] + ([LLM_CACHE_DIR / f"{key}.json"] if CACHE_MODE == "disk" else [])
    for path in paths:
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))["response"]
            except (ValueError, KeyError):
                continue
            CACHE_KEYS_USED.add(key)
            return data
    if key in _mem_cache:
        CACHE_KEYS_USED.add(key)
        return _mem_cache[key]
    return None


def _cache_put(key: str, data: dict, purpose: str) -> None:
    if CACHE_MODE != "disk":
        _mem_cache[key] = data
        return
    LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (LLM_CACHE_DIR / f"{key}.json").write_text(
        json.dumps({"response": data, "purpose": purpose, "model": LLM_MODEL}, ensure_ascii=False), encoding="utf-8")


def clear_runtime_cache() -> None:
    _mem_cache.clear()


def llm_enabled() -> bool:
    return os.environ.get("MOCK_LLM") != "1" and bool(os.environ.get("OPENAI_API_KEY"))


def llm_spent() -> float:
    """Total logged LLM spend across all runs (from the ledger)."""
    global _spent
    with _llm_lock:
        if _spent is None:
            _spent = 0.0
            if LEDGER.exists():
                for line in LEDGER.read_text(encoding="utf-8").splitlines():
                    try:
                        _spent += float(json.loads(line).get("cost", 0.0))
                    except ValueError:
                        pass
        return _spent


def _log_llm(entry: dict) -> None:
    global _spent
    llm_spent()
    with _llm_lock:
        _spent = (_spent or 0.0) + entry.get("cost", 0.0)
        CACHE_ROOT.mkdir(exist_ok=True)
        with open(LEDGER, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": utcnow_iso(), "model": LLM_MODEL, **entry}) + "\n")


def _client():
    global _llm_client
    with _llm_lock:
        if _llm_client is None:
            from openai import OpenAI
            _llm_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=0)
        return _llm_client


def llm_json(messages: list[dict], purpose: str, max_tokens: int = 500, deadline: Optional[float] = None) -> Optional[dict]:
    """JSON-mode chat completion. Cache first; None on any failure, budget cap or insufficient time.
    VERA_DISABLE_LLM=1 skips even the cache (fully deterministic template/rule behaviour for tests)."""
    if os.environ.get("VERA_DISABLE_LLM") == "1":
        return None
    key = hashlib.sha256(json.dumps([LLM_MODEL, max_tokens, messages], ensure_ascii=False, sort_keys=True)
                         .encode("utf-8")).hexdigest()
    cached = _cache_get(key)
    if cached is not None:
        return cached
    if not llm_enabled() or llm_spent() >= SPEND_CAP:
        return None
    timeout = LLM_TIMEOUT_S if deadline is None else min(LLM_TIMEOUT_S, deadline - time.monotonic() - 0.3)
    if timeout < 1.5:
        return None
    t0 = time.monotonic()
    try:
        resp = _client().chat.completions.create(
            model=LLM_MODEL, messages=messages, reasoning_effort="none", seed=7,
            response_format={"type": "json_object"}, max_completion_tokens=max_tokens, timeout=timeout)
    except Exception as e:
        _log_llm({"purpose": purpose, "error": type(e).__name__, "latency_s": round(time.monotonic() - t0, 2), "cost": 0.0})
        return None
    u = resp.usage
    cached = getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0
    _log_llm({"purpose": purpose, "in": u.prompt_tokens, "cached_in": cached, "out": u.completion_tokens,
              "latency_s": round(time.monotonic() - t0, 2),
              "cost": round(u.prompt_tokens / 1e6 * PRICE_IN + u.completion_tokens / 1e6 * PRICE_OUT, 7)})
    try:
        data = json.loads(resp.choices[0].message.content or "")
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    _cache_put(key, data, purpose)
    return data


# ---------------------------------------------------------------------------
# Fact sheet: the trimmed, jargon-free view of the 4 contexts the LLM composes from
# ---------------------------------------------------------------------------

CTAS = {"binary_yes_no", "binary_confirm_cancel", "multi_choice_slot", "open_ended", "none"}
SIGNAL_TEXT = {
    "ctr_below_peer_median": "click-through rate below the peer median", "unverified_gbp": "Google profile not verified",
    "no_active_offers": "no active offers", "stale_posts": "Google posts are stale",
    "engaged_in_last_48h": "replied to Vera in the last 48 hours", "engaged_in_last_24h": "replied to Vera in the last 24 hours",
    "renewal_due_soon": "subscription renewal due soon", "perf_dip_severe": "sharp performance drop",
    "above_peer_median_calls": "calls above the peer median", "growing_views_7d": "profile views growing this week",
    "perf_dip_post_expiry": "performance dropped after the plan lapsed", "ipl_eligible_locality": "locality with IPL-night demand",
    "seasonal_dip_apr_may": "in the usual April-May seasonal dip", "no_recent_post": "no recent Google post",
    "delivery_not_set_up": "home delivery not set up", "no_recent_conversation": "no recent conversation with Vera",
}
KIND_ANGLE = {
    "research_digest": "Lead with the source and the one finding most relevant to this business's patients/customers; offer to draft shareable content.",
    "regulation_change": "State the change, the deadline and what it means for them concretely; cite the source; offer a ready checklist.",
    "cde_opportunity": "Professional-development event: title, date/time, credits and fee from FACTS; offer to add it to their calendar.",
    "supply_alert": "Urgent but calm: batch numbers and bounded risk; offer to draft the customer note and replacement workflow.",
    "category_seasonal": "Quote the demand shifts and recommend one concrete shelf/offer move for this week.",
    "festival_upcoming": "Festival countdown; if it is months away frame it as early planning; tie to their own offer and the seasonal pattern.",
    "ipl_match_today": "Match today: state the real weekday from event details, use related_knowledge to judge whether a match-night promo helps (weeknight vs weekend) and recommend the better move.",
    "competitor_opened": "Competitor facts only from the event; contrast with their own offer/position (loss aversion) without disparaging anyone.",
    "perf_dip": "Name the exact drop and compare with peers; offer a quick diagnosis with fixes; calm, not alarmist. Name the time window of every number (last 7 days vs last 30 days).",
    "perf_spike": "Name the rise and its likely driver; suggest capitalising on the momentum. Name the time window of every number (last 7 days vs last 30 days).",
    "seasonal_perf_dip": "Reassure that the dip is seasonal using the seasonal pattern; redirect effort to retention of existing members/customers. Name the time window of every number.",
    "milestone_reached": "Celebrate the specific number and the gap to the milestone; offer a review-invite or thank-you post.",
    "review_theme_emerged": "Quote the theme and count (and a real quote if present); offer a public reply plus one fix they can announce.",
    "dormant_with_vera": "Re-open gently with one useful fact about their own numbers; no guilt; low-effort ask.",
    "curious_ask_due": "Ask the owner one easy question about their business this week (make a specific guess from FACTS); offer to turn the answer into a post. Open-ended.",
    "renewal_due": "Days left and amount; show what continues if they renew; no pressure tactics.",
    "winback_eligible": "Days since the plan lapsed and what has slipped since (loss aversion from FACTS); offer to restart.",
    "gbp_unverified": "Verification status, estimated uplift and the verification path; offer a guided walkthrough.",
    "active_planning_intent": "The owner already said yes and asked what it would look like: put the actual first outline in 'detail' (2-4 short parts: who it is for, format, pricing, how ordering/booking works). Use evidence from recent_conversation (e.g. current order volume) and, if present, planning_suggestions (label those numbers as suggestions for the owner to approve). Invent no other numbers. No qualifying questions; ask for approval.",
    "recall_due": "Customer recall: the service due, last visit and the open slots listed in FACTS; booking CTA with numbered slots.",
    "appointment_tomorrow": "Friendly reminder of tomorrow's appointment; confirm-or-reschedule CTA.",
    "chronic_refill_due": "Pharmacy: exact medicines and run-out date, applicable offers, delivery if saved; CONFIRM CTA. Other businesses: treat it as their regular follow-up visit being due.",
    "customer_lapsed_soft": "Warm, no-shame winback using the real gap and a live offer; single YES CTA.",
    "customer_lapsed_hard": "Warm, no-shame winback using the real gap, their past goal and a live offer; single YES CTA.",
    "trial_followup": "Thank them for the trial and offer the real next session; single YES CTA.",
    "wedding_package_followup": "Wedding countdown and the next-step window; offer to book the first session on their preferred day.",
}
DEFAULT_ANGLE = "Explain why now using only FACTS, anchor on this business's own numbers or offers, and propose one concrete next step."
EVENT_LABEL = {
    "research_digest": "new research relevant to their practice", "regulation_change": "a regulation change",
    "cde_opportunity": "an upcoming professional-education session", "supply_alert": "a product recall alert",
    "category_seasonal": "a seasonal demand shift", "festival_upcoming": "an upcoming festival",
    "ipl_match_today": "an IPL match in their city today", "competitor_opened": "a new competitor nearby",
    "perf_dip": "a recent drop in their Google profile results", "perf_spike": "a recent rise in their Google profile results",
    "seasonal_perf_dip": "a seasonal dip in their Google profile results", "milestone_reached": "a milestone on their listing",
    "review_theme_emerged": "a pattern in recent reviews", "dormant_with_vera": "no conversation with Vera for a while",
    "curious_ask_due": "a quick weekly question to the owner about their business",
    "renewal_due": "their magicpin plan renewal coming up", "winback_eligible": "their lapsed magicpin plan",
    "gbp_unverified": "their unverified Google profile", "active_planning_intent": "an idea the owner already said yes to",
    "recall_due": "a routine recall visit that is due", "appointment_tomorrow": "their appointment tomorrow",
    "chronic_refill_due": "a regular refill or follow-up that is due", "customer_lapsed_soft": "a gap since their last visit",
    "customer_lapsed_hard": "a long gap since their last visit", "trial_followup": "a follow-up after their trial session",
    "wedding_package_followup": "their upcoming wedding and pre-wedding prep",
}
REQUIRED_CONSENT = {"recall_due": ("recall_reminders",), "appointment_tomorrow": ("appointment_reminders",),
                    "chronic_refill_due": ("refill_reminders",), "trial_followup": ("program_updates", "kids_program_updates"),
                    "customer_lapsed_soft": ("winback_offers", "promotional_offers"),
                    "customer_lapsed_hard": ("winback_offers", "promotional_offers"),
                    "wedding_package_followup": ("bridal_package_followup",)}
KW_STOP = {"the", "and", "for", "with", "due", "today", "upcoming", "opened", "reached", "emerged", "new", "your", "this",
           "that", "from", "are", "was", "has", "have", "not", "all", "per", "you"}


def _signal_plain(sig: str) -> str:
    name, _, value = str(sig).partition(":")
    m = re.match(r"^(.*?)_(\d+)d$", name)
    if m:
        name, value = m.group(1), f"{m.group(2)}d"
    text = SIGNAL_TEXT.get(name, words(name))
    if value:
        d = re.match(r"^(\d+)d$", value)
        text += f" ({d.group(1)} days)" if d else f" ({words(value)})"
    return text


def _readable(key: str, value, weekday: bool = True):
    """Payload values made human-readable: percentages, dates, enum tokens, slot labels.
    Weekdays are added for event dates (they matter for timing) but not for history dates."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)) and key.endswith("_pct"):
        return f"{value * 100:+.0f}%"
    if isinstance(value, str):
        dt = parse_ts(value) if re.match(r"^\d{4}-\d{2}-\d{2}(T|$)", value) else None
        if dt:
            clock = "" if (dt.hour, dt.minute) == (0, 0) else f", {human_time(dt)}"
            day = f"{dt:%A} " if weekday else ""
            return f"{day}{dt.day} {dt:%b} {dt.year}{clock}"
        return re.sub(r"(\d+)_(month|day|week)", r"\1-\2", value).replace("_", " ") if "_" in value and " " not in value else value
    if isinstance(value, list):
        if value and all(isinstance(v, dict) and v.get("label") for v in value):
            return [v["label"] for v in value]
        return [_readable(key, v, weekday) for v in value]
    if isinstance(value, dict):
        return {words(re.sub(r"_iso$", "", k)): _readable(k, v, weekday) for k, v in value.items()}
    return value


def _pct_dict(d: dict) -> dict:
    return {words(k).replace(" pct", " %"): (round(v * 100, 1) if k.endswith("_pct") and isinstance(v, (int, float)) else v)
            for k, v in (d or {}).items() if v not in (None, "", [])}


def _trim_item(it: dict) -> dict:
    keep = {"title": it.get("title"), "source": it.get("source"), "summary": it.get("summary"),
            "what_to_do": it.get("actionable")}
    if it.get("trial_n"):
        keep["trial_size"] = it["trial_n"]
    if it.get("patient_segment"):
        keep["patient_segment"] = words(it["patient_segment"])
    if it.get("date"):
        keep["date"] = human_datetime(it["date"])
    if it.get("credits"):
        keep["credits"] = it["credits"]
    return {k: humanize_dates(v) if isinstance(v, str) else v for k, v in keep.items() if v}


def _event_months(c: Ctx) -> set[int]:
    months = set()
    for key, value in c.payload.items():
        if isinstance(value, str) and re.match(r"^\d{4}-\d{2}-\d{2}", value) and key != "expires_at":
            months.add(int(value[5:7]) - 1)
    months |= {MONTHS.index(t) for t in str(c.payload.get("season_note", "")).lower().split("_") if t in MONTHS}
    return months


def _related_knowledge(c: Ctx) -> list[dict]:
    text_values = [words(c.kind)] + [words(v) for v in c.payload.values() if isinstance(v, str)]
    kws = {t for t in re.findall(r"[a-z]{3,}", " ".join(text_values).lower()) if t not in KW_STOP}
    scored = []
    for it in c.category.get("digest") or []:
        if c.item and it.get("id") == c.item.get("id"):
            continue
        blob_words = set(re.findall(r"[a-z]{3,}", f"{it.get('title', '')} {it.get('summary', '')}".lower()))
        score = sum(1 for k in kws if any(w.startswith(k) for w in blob_words))
        if score:
            scored.append((score, it))
    out = [{"kind": "insight", **_trim_item(it)} for _, it in sorted(scored, key=lambda x: -x[0])[:2]]
    months = _event_months(c)
    for beat in c.category.get("seasonal_beats") or []:
        if months & months_in(beat.get("month_range", "")):
            out.append({"kind": "seasonal pattern", "months": beat["month_range"], "note": beat["note"]})
    return out


def _language_mode(c: Ctx, customer_facing: bool) -> tuple[str, str]:
    if customer_facing:
        pref = (((c.customer or {}).get("identity") or {}).get("language_pref") or "").lower()
        if pref == "hi":
            return "hindi_first", "Hindi-first Hinglish in Latin script (respectful 'aap'); medicine/product names stay in English."
        if pref.startswith("hi"):
            return "hinglish", "Natural Hinglish (Hindi-English code-mix in Latin script), e.g. 'Aapke liye 2 slots ready hain'."
        return "english", "Clear, warm English."
    # Owners get the language they actually write in: Hinglish only if their own messages are Hinglish.
    own = " ".join(t.get("body", "") for t in c.merchant.get("conversation_history") or [] if t.get("from") == "merchant")
    if own and "hi" in (c.ident.get("languages") or []) and conversation_handlers.detect_lang(own) == "hi":
        return "hinglish", "Natural Hinglish (Hindi-English code-mix in Latin script), matching how this owner writes."
    return "english", "Clear, natural English (the owner writes in English)."


def _planning_suggestions(c: Ctx) -> list[str]:
    """Bulk-pricing tiers derived from the merchant's own offer price (Vera recommends pricing; these are proposals)."""
    topic = str(c.payload.get("intent_topic") or "")
    if c.kind != "active_planning_intent" or not re.search(r"bulk|corporate|group|office", topic) or not c.offers:
        return []
    m = re.search(r"₹\s?([\d,]+)", c.offers[0])
    if not m:
        return []
    price = int(m.group(1).replace(",", ""))
    tiers = [(qty, int(round(price * (1 - disc) / 5) * 5)) for qty, disc in ((10, 0.10), (25, 0.15))]
    return [f"suggested tier: {qty}+ orders at {inr(p)} each (about {round((1 - p / price) * 100)}% below the "
            f"{inr(price)} list price)" for qty, p in tiers]


def build_facts(c: Ctx, customer_facing: bool, base: dict) -> dict:
    voice = c.category.get("voice") or {}
    peer = c.peer
    business = {
        "name": c.name, "owner_first_name": re.sub(r"^Dr\.?\s*", "", c.ident.get("owner_first_name") or "") or None,
        "locality": c.ident.get("locality"), "city": c.ident.get("city"),
        "google_profile_verified": c.ident.get("verified"),
        "subscription": _readable("subscription", {k: v for k, v in c.sub.items() if v not in (None, "")}, weekday=False),
        "last_30_days": {"profile views": c.perf.get("views"), "calls": c.perf.get("calls"),
                         "direction requests": c.perf.get("directions"), "leads": c.perf.get("leads"),
                         "click-through rate %": round(c.perf["ctr"] * 100, 1) if c.perf.get("ctr") is not None else None},
        "change_vs_previous_week_%": {words(k[:-4]): round(v * 100) for k, v in (c.perf.get("delta_7d") or {}).items()
                                      if isinstance(v, (int, float))},
        "active_offers": c.offers,
        "expired_offers": [o["title"] for o in c.merchant.get("offers") or [] if o.get("status") != "active" and o.get("title")],
        "customer_base": _pct_dict(c.agg),
        "account_notes": [_signal_plain(s) for s in c.merchant.get("signals") or []],
        "review_themes": [{k: v for k, v in {"theme": words(t.get("theme")), "sentiment": t.get("sentiment"),
                                             "mentions_30d": t.get("occurrences_30d"), "quote": t.get("common_quote")}.items() if v}
                          for t in c.merchant.get("review_themes") or []],
        "recent_conversation": [{"from": t.get("from"), "text": t.get("body")}
                                for t in (c.merchant.get("conversation_history") or [])[-3:]],
    }
    business["last_30_days"] = {k: v for k, v in business["last_30_days"].items() if v is not None}
    if customer_facing:
        # the customer is the reader: owner-only data (and the owner's name) only confuses who is being addressed
        business = {k: business[k] for k in ("name", "locality", "city", "active_offers")}
    facts = {
        "category": {
            "type": c.category.get("display_name") or c.slug,
            "voice": {"tone": words(voice.get("tone")), "register": words(voice.get("register")),
                      "allowed_vocabulary": (voice.get("vocab_allowed") or [])[:12],
                      "taboo_words": [re.sub(r"\s*\(.*\)$", "", t) for t in voice.get("vocab_taboo") or []]},
            "peer_benchmarks_30d": {"profile views": peer.get("avg_views_30d"), "calls": peer.get("avg_calls_30d"),
                                    "click-through rate %": round(peer["avg_ctr"] * 100, 1) if peer.get("avg_ctr") else None,
                                    "rating": peer.get("avg_rating"), "review count": peer.get("avg_review_count"),
                                    "scope": words(peer.get("scope"))},
        },
        "business": business,
        "event": {"what_happened": EVENT_LABEL.get(c.kind, words(c.kind)), "source": c.trigger.get("source"),
                  "urgency_1_to_5": c.trigger.get("urgency"),
                  "details_available": not c.placeholder,
                  "details": {} if c.placeholder else _readable("payload", {k: v for k, v in c.payload.items()
                                                                            if k not in ("top_item_id", "digest_item_id", "alert_id")})},
    }
    facts["category"]["peer_benchmarks_30d"] = {k: v for k, v in facts["category"]["peer_benchmarks_30d"].items() if v}
    if c.item:
        facts["research_item"] = _trim_item(c.item)
    related = _related_knowledge(c)
    if related:
        facts["related_knowledge"] = related
    suggestions = _planning_suggestions(c)
    if suggestions:
        facts["planning_suggestions"] = suggestions
    if not c.offers and not customer_facing:  # customers may only be offered what the business actually runs
        ideas = [o["title"] for o in c.category.get("offer_catalog") or [] if o.get("type") in ("service_at_price", "free_service")][:3]
        if ideas:
            facts["category_offer_ideas_not_yet_offered"] = ideas
    lang_mode, lang_text = _language_mode(c, customer_facing)
    instructions = {"language": lang_text, "angle": KIND_ANGLE.get(c.kind, DEFAULT_ANGLE),
                    "suggested_cta": base["cta"]}
    if customer_facing:
        greet, poss = c.cust_names()
        cust = c.customer or {}
        ident, rel, prefs = cust.get("identity") or {}, cust.get("relationship") or {}, cust.get("preferences") or {}
        facts["customer"] = {
            "name": ident.get("name"), "age_band": ident.get("age_band"), "state": words(cust.get("state")),
            "relationship": _readable("relationship", {k: v for k, v in rel.items() if v not in (None, [], "")}, weekday=False),
            "preferences": _readable("preferences", prefs, weekday=False),
            "consent_scope": [words(s) for s in (cust.get("consent") or {}).get("scope") or []],
        }
        instructions["send_as"] = (f"Write AS {c.name} to their customer, in first-person plural ('we'), e.g. "
                                   f"'{c.name} here'. Warm, respectful, no medical claims, no guilt.")
        instructions["opening_must_start_with"] = f"{greet}, {c.name} here."
        instructions["reader"] = f"The reader is the customer ({(cust.get('identity') or {}).get('name')}), not the business owner."
        if poss != "your":
            instructions["reader"] = (f"A family member reads this on behalf of {poss[:-2]}; refer to the service as "
                                      f"{poss[:-2]}'s (Hinglish: '{poss[:-2]} ki/ke').")
        needed = REQUIRED_CONSENT.get(c.kind)
        scope = set((cust.get("consent") or {}).get("scope") or [])
        if needed and not scope & set(needed) and "promotional_offers" in scope:
            instructions["consent_note"] = ("Keep it a light, friendly offer-style note rather than a formal reminder. "
                                            "Never mention consent, opt-ins or what kind of message this is.")
    else:
        instructions["send_as"] = "You are Vera writing to the business owner (peer to peer)."
        instructions["opening_must_start_with"] = f"{c.greet},"
    if c.placeholder:
        instructions["event_note"] = ("Only the kind of event is known. Mention it generally, invent no specifics, and never "
                                      "tie it to a particular number (don't say which figure the milestone, dip or spike is "
                                      "about). Anchor on the business's own data; never say that details are missing.")
    facts["instructions"] = instructions
    facts["reference_draft"] = base["body"]
    return _prune(facts)


def _prune(value):
    if isinstance(value, dict):
        out = {k: _prune(v) for k, v in value.items()}
        return {k: v for k, v in out.items() if v not in (None, "", [], {})}
    if isinstance(value, list):
        return [_prune(v) for v in value if v not in (None, "", [], {})]
    return value


# ---------------------------------------------------------------------------
# Prompt + validation
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are Vera, magicpin's WhatsApp assistant for Indian local businesses (dentists, salons, restaurants, gyms, pharmacies). Write ONE WhatsApp message from the FACTS JSON the user sends.

Hard rules (a draft that breaks any of them is rejected):
1. Facts only. Every number, price, date, time, name, source, percentage and claim must come from FACTS. Never invent competitors, research, statistics, prices, slots, customer counts, deadlines or delivery times. If a detail is missing, write around it.
2. Why now: the opening names the specific event behind this message (FACTS.event), in plain words.
3. One ask, placed last: exactly one low-friction call to action ("Reply YES", one yes/no question, or numbered slots for a booking). Never offer several different actions.
4. Voice: match category.voice (tone, register); use allowed vocabulary where natural; never use taboo words or hype ("guaranteed", "best in city", "amazing deal", "miracle"). Dentists and pharmacies: clinical-peer, precise, no medical claims.
5. Plain words: no URLs, hashtags, markdown, snake_case or internal terms (trigger, payload, signal, suppression, digest). No emojis in messages to business owners; at most one in customer messages.
6. Concise: 2-4 sentences, about 250-450 characters, never over 600. No preamble ("Hope you're doing well"), no self-introduction, never repeat business.recent_conversation.
7. Language: follow instructions.language exactly. When Vera writes Hindi she uses feminine forms ("kar rahi hoon", "bhej dungi"). Customer messages speak as the business ("hum", "we").
8. Own wording: improve on reference_draft (it is fact-checked) but do not copy it or any known example verbatim. Never copy text from instructions into the message.
9. The ask is short and says the next step once (e.g. "Want me to draft it? Reply YES."); do not restate what the detail already offered.
10. Never mention internal mechanics: consent, opt-ins, message types, scheduled check-ins, or how this message was triggered.

What scores well:
- One or two specific, checkable anchors: their numbers vs peer benchmarks, their offer price, a source with page/date, a count or deadline.
- Personal: the opening starts exactly with instructions.opening_must_start_with; tie the fact to THIS business (offers, cohort, locality, recent conversation) or, for customers, to their own history and preferences.
- Judgment: if related_knowledge changes the best move, recommend the better move in one line.
- Compulsion: one or two of loss aversion, curiosity, social proof (only from FACTS), effort externalisation ("I'll draft it"), reciprocity, asking the owner a question.
- Service + price beats "% off". Prefer the business's active offers; category_offer_ideas_not_yet_offered are suggestions, never claimed as theirs.

Return JSON only:
{"opening": "salutation + why-now sentence", "detail": "1-2 sentences with the anchor facts, relevance and judgment", "ask": "the single call-to-action sentence", "cta": "binary_yes_no | binary_confirm_cancel | multi_choice_slot | open_ended | none", "rationale": "max 30 words: facts used, lever, why now"}"""

URL_RX = re.compile(r"https?://|www\.|\.com\b", re.I)
SNAKE_RX = re.compile(r"\b[a-z0-9]+_[a-z0-9_]+\b")
JARGON_RX = re.compile(r"\b(payload|suppression|placeholder|digest item|trigger id|reference draft|curious ask|promotional"
                       r"|opted[ -]in|opt-in|consent|check-in message|perf(ormance)? (dip|spike) (is showing|flagged))\b", re.I)
EMOJI_RX = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
HYPE = ["guaranteed", "best in city", "amazing deal", "miracle", "100% safe"]
GENERIC_NAME_WORDS = {"dental", "clinic", "care", "salon", "studio", "pharmacy", "medicos", "fitness", "cafe", "restaurant",
                      "gym", "beauty", "health", "hair", "family", "plus", "express", "centre", "center", "house", "lounge",
                      "spa", "yoga", "the", "and", "junction", "diner", "point", "bar", "smile", "bright"}


def _load_case_study_bodies() -> list[str]:
    path = Path(__file__).with_name("examples") / "case-studies.md"
    if not path.exists():
        return []
    blocks = re.findall(r"```\n(.*?)\n```", path.read_text(encoding="utf-8"), re.S)
    return [" ".join(b.split()).lower() for b in blocks]


CASE_STUDY_BODIES = _load_case_study_bodies()


def _num_key(token: str) -> Optional[str]:
    try:
        return f"{float(token.replace(',', '')):g}"
    except ValueError:
        return None


def _allowed_numbers(*texts: str) -> set[str]:
    allowed = set()
    for text in texts:
        for tok in re.findall(r"\d[\d,]*(?:\.\d+)?", text):
            key = _num_key(tok)
            if key is None:
                continue
            allowed.add(key)
            value = float(key)
            if 0 < abs(value) < 1:
                allowed |= {f"{round(abs(value) * 100):g}", f"{round(abs(value) * 100, 1):g}"}
    return allowed


def validate_llm(resp: dict, c: Ctx, facts: dict, base: dict, customer_facing: bool) -> tuple[list[str], str]:
    parts = [resp.get(k) for k in ("opening", "detail", "ask")]
    if not all(isinstance(p, str) and p.strip() for p in parts):
        return ["opening, detail and ask must all be non-empty strings"], ""
    body = render("{{1}} {{2}} {{3}}", [p.strip() for p in parts])
    low = body.lower()
    problems = []
    if resp.get("cta") not in CTAS:
        problems.append(f"cta must be one of {sorted(CTAS)}")
    if not 90 <= len(body) <= 650:
        problems.append(f"the message is {len(body)} characters; keep it between 250 and 450")
    if URL_RX.search(body):
        problems.append("remove the URL")
    for rx in (SNAKE_RX, JARGON_RX):
        m = rx.search(body)
        if m:
            problems.append(f"remove the internal term '{m.group(0)}'")
    taboo = [t.lower() for t in (facts.get("category", {}).get("voice", {}).get("taboo_words") or [])] + HYPE
    problems += [f"remove the taboo phrase '{t}'" for t in taboo if t and t in low]
    allowed = _allowed_numbers(json.dumps(facts, ensure_ascii=False), base["body"])
    bad = sorted({tok for tok in re.findall(r"\d[\d,]*(?:\.\d+)?", body)
                  if (_num_key(tok) or "0") not in allowed and float(_num_key(tok) or 0) > 10})
    if bad:
        problems.append(f"these numbers are not in FACTS: {', '.join(bad)}; remove them or use exact FACTS figures")
    if len(EMOJI_RX.findall(body)) > (1 if customer_facing else 0):
        problems.append("remove the emojis" if not customer_facing else "use at most one emoji")
    prefix = facts.get("instructions", {}).get("opening_must_start_with", "")
    if prefix and not low.startswith(prefix.split(",")[0].lower()):
        problems.append(f"the opening must start exactly with '{prefix}'")
    if re.search(r"(don'?t|do not) have (the |any )?(\w+ )?details|details (are |were )?(not available|unavailable|missing)"
                 r"|no (further )?details", low):
        problems.append("never mention missing details; write around them")
    if customer_facing:
        greet, poss = c.cust_names()
        who = greet.split(" ", 1)[1] if " " in greet else poss.replace("'s", "").replace(" ji", "")
        if who and who != "your" and who.split()[0].lower() not in low:
            problems.append(f"address the customer by name ({who})")
        distinctive = [t for t in re.findall(r"[a-z0-9]+", c.name.lower()) if len(t) >= 3 and t not in GENERIC_NAME_WORDS]
        if distinctive and distinctive[0] not in low:
            problems.append(f"say who is writing: the message is from {c.name}")
        if not c.offers and re.search(r"\b(offers?|discount|deal)\b", low):
            problems.append("the business has no active offer: do not mention or imply any offer or discount")
    if re.search(r"reads this|family member reads|instructions|reference draft", low):
        problems.append("do not copy instruction text into the message")
    lang_mode, _ = _language_mode(c, customer_facing)
    markers = sum(t in conversation_handlers.HINGLISH_MARKERS for t in re.findall(r"[a-z]+", low))
    need = {"hindi_first": 3, "hinglish": 2}.get(lang_mode, 0)
    if markers < need:
        problems.append("follow instructions.language: add natural Hindi-English code-mix (Latin script)")
    if resp.get("cta") != "none" and not re.search(r"\?|\breply\b|\bbatayein\b|\bbata dijiye\b", parts[2], re.I):
        problems.append("the ask must end with one clear call to action (a question or 'Reply ...')")
    history = [t.get("body", "") for t in c.merchant.get("conversation_history") or [] if t.get("from") == "vera"]
    if any(difflib.SequenceMatcher(None, low, h.lower()).ratio() > 0.8 for h in history if h):
        problems.append("too similar to a message already sent to this business; say something new")
    if any(difflib.SequenceMatcher(None, low, cs).ratio() > 0.6 for cs in CASE_STUDY_BODIES):
        problems.append("too close to a published example; use your own wording")
    return problems, body


def _compose_llm(c: Ctx, base: dict, deadline: Optional[float]) -> dict:
    customer_facing = base["send_as"] == "merchant_on_behalf"
    facts = build_facts(c, customer_facing, base)
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "FACTS:\n" + json.dumps(facts, ensure_ascii=False, separators=(",", ":"))}]
    purpose = f"compose:{c.kind}"
    resp = llm_json(messages, purpose, deadline=deadline)
    if resp is None:
        return {"composer": "none", "problems": ["LLM unavailable, over budget or out of time"]}
    problems, body = validate_llm(resp, c, facts, base, customer_facing)
    if problems and (deadline is None or deadline - time.monotonic() > 4):
        retry = messages + [{"role": "assistant", "content": json.dumps(resp, ensure_ascii=False)},
                            {"role": "user", "content": "Fix these problems and return the corrected JSON only:\n- "
                                                        + "\n- ".join(problems)}]
        resp2 = llm_json(retry, purpose + ":retry", deadline=deadline)
        if resp2 is not None:
            resp, (problems, body) = resp2, validate_llm(resp2, c, facts, base, customer_facing)
    if problems:
        return {"composer": "rejected", "problems": problems}
    params = [resp[k].strip() for k in ("opening", "detail", "ask")]
    return {
        **base,
        "body": body,
        "cta": resp["cta"],
        "rationale": str(resp.get("rationale") or base["rationale"]).strip()[:300],
        "template_name": "merchant_compose_v1" if customer_facing else "vera_compose_v1",
        "template_params": params,
        "composer": "llm",
    }


# ---------------------------------------------------------------------------
# LLM replies: conversation_handlers decides the action; this only writes the text (with rule fallback)
# ---------------------------------------------------------------------------

REPLY_BUDGET_S = float(os.environ.get("VERA_REPLY_BUDGET_S", "8"))

REPLY_SYSTEM_PROMPT = """You continue a WhatsApp conversation for magicpin's assistant, Vera. FACTS describe the business (and the customer, if any). CONVERSATION is the thread so far ("you" = your earlier messages). TASK says what this reply must do.

Hard rules (a reply that breaks any of them is rejected):
1. Facts only: every number, price, date, time, name and claim must come from FACTS or CONVERSATION. If the answer is not there, say briefly that you'll check and confirm; never guess.
2. Do exactly what TASK.what_to_do says, in TASK.language. When Vera writes Hindi she uses feminine forms ("kar rahi hoon", "bhej dungi"). In customer conversations you speak as the business ("hum", "we").
3. No greeting, no self-introduction, never repeat or closely paraphrase an earlier message.
4. End with one next step matching TASK.cta. After the other person has said yes, never ask qualifying questions ("would you", "do you", "shall I", "what if", "how about").
5. Plain text: no URLs, attachments, markdown headers, hashtags, snake_case or internal terms (trigger, payload, signal, consent, opt-in). Drafts go inline as short lines.
6. Concise: answers 1-3 sentences; a delivered draft may use up to about 8 short lines.

Return JSON only: {"body": "the reply", "cta": "binary_yes_no | binary_confirm_cancel | multi_choice_slot | open_ended | none", "rationale": "max 25 words"}"""

REPLY_TASKS = {
    "action": ("The other person just said yes. Deliver the promised work NOW, inline: write the actual draft (post, "
               "WhatsApp note, checklist or outline) from FACTS in a few short lines, then ask them to reply CONFIRM or "
               "send edits. Do not ask about their preferences. Copy meant for customers or patients must be ready to "
               "send: in the business's voice, naming the business, with one call to action and no internal notes or "
               "caveats. It may only promote services and offers the business actually lists (business.active_offers); "
               "anything else must be labelled as a suggestion for the owner to approve."),
    "question": ("Answer the owner's latest question directly from FACTS/CONVERSATION in 1-3 sentences. If FACTS don't "
                 "cover it, say so briefly and that you'll check. Then offer the one next step from the conversation."),
    "engaged": "Acknowledge specifically what the owner just said, adapt the plan to it using FACTS, and offer one next step.",
    "customer_question": ("Answer the customer's latest message as the business ('we'/'hum', never 'I'/'main'), from FACTS "
                          "only (offers, open slots, their history). If the answer isn't in FACTS, say we'll confirm "
                          "shortly. End with the booking or confirmation next step (e.g. the open slots to reply with)."),
}
QUALIFYING_RX = re.compile(r"\b(would you|do you|shall i|should i|can you tell|what if|how about|may i)\b", re.I)
REINTRO_RX = re.compile(r"\b(i am|i'm|this is) vera\b|\bvera here\b|^(hi|hello|namaste|dear)\b", re.I)
ATTACH_RX = re.compile(r"\b(attached|attachment|pdf|download)\b", re.I)
FIRST_PERSON_RX = re.compile(r"\b(i'll|i will|i am|i'm|i can|main|mujhe|dungi|karungi|rahi hoon|bataungi)\b", re.I)


def validate_reply(resp: dict, intent: str, facts: dict, conv: dict, reference: str, lang: str,
                   customer_facing: bool) -> tuple[list[str], str]:
    body = " ".join(str(resp.get("body") or "").split()) if intent != "action" else str(resp.get("body") or "").strip()
    if not body:
        return ["body must be a non-empty string"], ""
    low = body.lower()
    problems = []
    if resp.get("cta") not in CTAS:
        problems.append(f"cta must be one of {sorted(CTAS)}")
    limit = 900 if intent == "action" else 450
    if not 30 <= len(body) <= limit:
        problems.append(f"the reply is {len(body)} characters; keep it under {limit}")
    for rx, what in ((URL_RX, "URL"), (SNAKE_RX, "internal term"), (JARGON_RX, "internal term"), (ATTACH_RX, "attachment")):
        m = rx.search(body)
        if m:
            problems.append(f"remove the {what} '{m.group(0)}'")
    if REINTRO_RX.search(body):
        problems.append("no greeting or self-introduction mid-conversation")
    taboo = [t.lower() for t in (facts.get("category", {}).get("voice", {}).get("taboo_words") or [])] + HYPE
    problems += [f"remove the taboo phrase '{t}'" for t in taboo if t and t in low]
    convo_text = " ".join(t.get("body", "") for t in conv.get("turns") or [])
    allowed = _allowed_numbers(json.dumps(facts, ensure_ascii=False), convo_text, reference)
    bad = sorted({tok for tok in re.findall(r"\d[\d,]*(?:\.\d+)?", body)
                  if (_num_key(tok) or "0") not in allowed and float(_num_key(tok) or 0) > 10})
    if bad:
        problems.append(f"these numbers are not in FACTS or CONVERSATION: {', '.join(bad)}; remove them")
    if len(EMOJI_RX.findall(body)) > 1:
        problems.append("use at most one emoji")
    if customer_facing and FIRST_PERSON_RX.search(body):
        problems.append(f"speak as the business ('we'/'hum'), not '{FIRST_PERSON_RX.search(body).group(0)}'")
    if intent == "action" and QUALIFYING_RX.search(body):
        problems.append(f"the person already said yes: remove the question '{QUALIFYING_RX.search(body).group(0)}'")
    if resp.get("cta") != "none" and not re.search(r"\?|\breply\b|\bconfirm\b|\bbatayein\b|\bbata dijiye\b", low):
        problems.append("end with one clear next step (a question, 'Reply ...' or 'CONFIRM')")
    markers = sum(t in conversation_handlers.HINGLISH_MARKERS for t in re.findall(r"[a-z]+", low))
    if lang == "hi" and markers < 2:
        problems.append("reply in natural Hinglish (Latin script), matching the other person's language")
    earlier = [t.get("body", "").lower() for t in conv.get("turns") or [] if t.get("from") == "bot"]
    if any(difflib.SequenceMatcher(None, low, e).ratio() > 0.8 for e in earlier if e):
        problems.append("too similar to an earlier message; say something new")
    return problems, body


def write_reply(state: dict, intent: str, reference: str, cta: str, lang: str, deadline: float) -> Optional[dict]:
    """LLM-written reply text for an action already chosen by conversation_handlers; None -> keep the rule text."""
    merchant, category = state.get("merchant"), state.get("category")
    if not merchant or not category or intent not in REPLY_TASKS:
        return None
    conv = state.get("conversation") or {}
    trigger = state.get("trigger") or {"kind": conv.get("kind") or "", "payload": {}}
    customer = state.get("customer") if conv.get("send_as") == "merchant_on_behalf" else None
    customer_facing = customer is not None
    base = compose_template(category, merchant, trigger, customer)
    facts = build_facts(Ctx(category, merchant, trigger, customer), customer_facing, base)
    facts.pop("instructions", None)
    facts.pop("reference_draft", None)
    turns = [{"from": "you" if t.get("from") == "bot" else t.get("from"), "text": t.get("body")}
             for t in (conv.get("turns") or [])[-8:]]
    task = {"what_to_do": REPLY_TASKS[intent],
            "language": ("Natural Hinglish (Hindi-English code-mix, Latin script), matching their last message."
                         if lang == "hi" else "Clear, natural English."),
            "cta": cta, "reference_reply": reference,
            "speaker": (f"You write as {merchant.get('identity', {}).get('name')} to their customer." if customer_facing
                        else "You are Vera writing to the business owner.")}
    user = ("FACTS:\n" + json.dumps(facts, ensure_ascii=False, separators=(",", ":"))
            + "\n\nCONVERSATION:\n" + json.dumps(turns, ensure_ascii=False, separators=(",", ":"))
            + "\n\nTASK:\n" + json.dumps(task, ensure_ascii=False, separators=(",", ":")))
    messages = [{"role": "system", "content": REPLY_SYSTEM_PROMPT}, {"role": "user", "content": user}]
    resp = llm_json(messages, f"reply:{intent}", max_tokens=700, deadline=deadline)
    if resp is None:
        return None
    problems, body = validate_reply(resp, intent, facts, conv, reference, lang, customer_facing)
    if problems and deadline - time.monotonic() > 4:
        retry = messages + [{"role": "assistant", "content": json.dumps(resp, ensure_ascii=False)},
                            {"role": "user", "content": "Fix these problems and return the corrected JSON only:\n- "
                                                        + "\n- ".join(problems)}]
        resp2 = llm_json(retry, f"reply:{intent}:retry", max_tokens=700, deadline=deadline)
        if resp2 is not None:
            resp, (problems, body) = resp2, validate_reply(resp2, intent, facts, conv, reference, lang, customer_facing)
    if problems:
        REPLY_REJECTIONS.append({"intent": intent, "problems": problems})
        return None
    return {"body": body, "cta": resp["cta"], "rationale": str(resp.get("rationale") or "").strip()[:200]}


REPLY_REJECTIONS: list[dict] = []  # diagnostics for tools/reply_eval.py


# ---------------------------------------------------------------------------
# Tick: decide what (if anything) to send
# ---------------------------------------------------------------------------

def _trigger_refs(trg: dict) -> tuple[Optional[str], Optional[str]]:
    payload = trg.get("payload") or {}
    return trg.get("merchant_id") or payload.get("merchant_id"), trg.get("customer_id") or payload.get("customer_id")


def _within(ts: Optional[str], now: datetime, window: timedelta) -> bool:
    dt = parse_ts(ts)
    return dt is not None and now - dt < window


def _eligible(tid: str, trg: dict, now: datetime) -> bool:
    sup = trg.get("suppression_key") or tid
    if sup in store.used_suppression:
        return False
    if STRICT_EXPIRY and (parse_ts(trg.get("expires_at")) or FAR_FUTURE) < now:
        return False
    merchant_id, customer_id = _trigger_refs(trg)
    merchant = store.find("merchant", merchant_id)
    if not merchant or not store.find("category", merchant.get("category_slug")):
        return False
    ms = store.mstate(merchant["merchant_id"])
    for key in ("opted_out_until", "wait_until"):
        until = parse_ts(ms.get(key))
        if until and until > now:
            return False
    if trg.get("scope") == "customer":
        customer = store.find("customer", customer_id)
        if not customer or not consent_ok(customer, trg.get("kind", "")):
            return False
        cs = store.cstate(customer["customer_id"])
        return not cs["opted_out"] and not _within(cs["last_activity_at"], now, COOLDOWN)
    if ms["unanswered"] >= MAX_UNANSWERED:
        return False
    urgent = int(trg.get("urgency") or 0) >= URGENT
    return urgent or not _within(ms["last_activity_at"], now, COOLDOWN)


def _new_conversation_id(tid: str) -> str:
    base = "conv_" + re.sub(r"[^A-Za-z0-9_]+", "_", tid)
    cid, n = base, 2
    while cid in store.conversations:
        cid, n = f"{base}_{n}", n + 1
    return cid


def run_tick(now_iso: Optional[str], available: list[str]) -> list[dict]:
    now = parse_ts(now_iso) or datetime.now(timezone.utc)
    with store.lock:
        candidates = []
        for tid in dict.fromkeys(available or []):
            trg = store.find("trigger", tid)
            if trg and _eligible(tid, trg, now):
                candidates.append((tid, trg))
        candidates.sort(key=lambda x: (-int(x[1].get("urgency") or 0), parse_ts(x[1].get("expires_at")) or FAR_FUTURE))
        chosen, targets = [], set()
        for tid, trg in candidates:
            merchant_id, customer_id = _trigger_refs(trg)
            target = ("customer", customer_id) if trg.get("scope") == "customer" else ("merchant", merchant_id)
            if target in targets:
                continue
            targets.add(target)
            chosen.append((tid, trg))
            store.used_suppression.add(trg.get("suppression_key") or tid)
            if len(chosen) == MAX_ACTIONS_PER_TICK:
                break

    deadline = time.monotonic() + TICK_BUDGET_S
    jobs = []
    for tid, trg in chosen:
        merchant_id, customer_id = _trigger_refs(trg)
        merchant = store.find("merchant", merchant_id)
        category = store.find("category", merchant.get("category_slug"))
        customer = store.find("customer", customer_id) if trg.get("scope") == "customer" else None
        jobs.append((tid, trg, merchant, category, customer, EXECUTOR.submit(compose, category, merchant, trg, customer, deadline)))
    wait([job[-1] for job in jobs], timeout=TICK_BUDGET_S + 0.5)

    actions = []
    for tid, trg, merchant, category, customer, fut in jobs:
        try:
            msg = fut.result(timeout=0) if fut.done() else compose_template(category, merchant, trg, customer)
        except Exception:
            try:
                msg = compose_template(category, merchant, trg, customer)
            except Exception:
                with store.lock:
                    store.used_suppression.discard(trg.get("suppression_key") or tid)
                continue
        actions.append((tid, trg, merchant, customer, msg))

    out = []
    with store.lock:
        for tid, trg, merchant, customer, msg in actions:
            conv_id = _new_conversation_id(tid)
            store.conversations[conv_id] = {
                "conversation_id": conv_id, "merchant_id": merchant["merchant_id"],
                "customer_id": customer["customer_id"] if customer else None, "trigger_id": tid,
                "kind": trg.get("kind"), "send_as": msg["send_as"], "status": "open", "origin": "tick",
                "turns": [{"from": "bot", "body": msg["body"], "ts": to_iso(now)}], "sent_bodies": [msg["body"]],
                "started_at": to_iso(now), "last_activity": to_iso(now), "wait_until": None,
            }
            if customer:
                store.cstate(customer["customer_id"])["last_activity_at"] = to_iso(now)
            else:
                ms = store.mstate(merchant["merchant_id"])
                ms["last_activity_at"] = to_iso(now)
                ms["unanswered"] += 1
            out.append({
                "conversation_id": conv_id,
                "merchant_id": merchant["merchant_id"],
                "customer_id": customer["customer_id"] if customer else None,
                "send_as": msg["send_as"],
                "trigger_id": tid,
                "template_name": msg["template_name"],
                "template_params": msg["template_params"],
                "body": msg["body"],
                "cta": msg["cta"],
                "suppression_key": msg["suppression_key"],
                "rationale": msg["rationale"],
            })
    return out


# ---------------------------------------------------------------------------
# Reply: record the inbound turn, delegate the decision, enforce invariants
# ---------------------------------------------------------------------------

class TickBody(BaseModel):
    now: Optional[str] = None
    available_triggers: list[str] = []


class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str = "merchant"
    message: str = ""
    received_at: Optional[str] = None
    turn_number: Optional[int] = None


def handle_reply(body: ReplyBody) -> dict:
    now = parse_ts(body.received_at) or datetime.now(timezone.utc)
    deadline = time.monotonic() + REPLY_BUDGET_S
    with store.lock:
        conv = store.conversations.get(body.conversation_id)
        if conv is None:
            conv = store.conversations[body.conversation_id] = {
                "conversation_id": body.conversation_id, "merchant_id": body.merchant_id,
                "customer_id": body.customer_id, "trigger_id": None, "kind": None,
                "send_as": "merchant_on_behalf" if body.from_role == "customer" else "vera",
                "status": "open", "origin": "inbound", "turns": [], "sent_bodies": [],
                "started_at": to_iso(now), "last_activity": to_iso(now), "wait_until": None,
            }
        if not conv.get("merchant_id") and body.merchant_id:
            conv["merchant_id"] = body.merchant_id
        if conv["status"] == "ended":
            return {"action": "end", "rationale": "Conversation already closed; not sending anything further."}
        conv["status"] = "open"
        conv["turns"].append({"from": body.from_role, "body": body.message, "ts": to_iso(now)})
        conv["last_activity"] = to_iso(now)
        merchant = store.find("merchant", conv.get("merchant_id"))
        if conv.get("merchant_id"):
            store.mstate(conv["merchant_id"])["last_activity_at"] = to_iso(now)
        sender = store.sender_state(conv, body.from_role)
        normalized = re.sub(r"\s+", " ", body.message.strip().lower())
        sender["inbound_repeat"] = sender["inbound_repeat"] + 1 if normalized and normalized == sender["last_inbound"] else 1
        sender["last_inbound"] = normalized
        state = {
            "conversation": conv,
            "merchant": merchant,
            "category": store.find("category", (merchant or {}).get("category_slug")),
            "customer": store.find("customer", conv.get("customer_id")),
            "trigger": store.find("trigger", conv.get("trigger_id")),
            "merchant_state": dict(store.mstate(conv["merchant_id"])) if conv.get("merchant_id") else {},
            "sender_state": dict(sender),
            "from_role": body.from_role,
            "turn_number": body.turn_number,
            "now": to_iso(now),
        }
        state["write_reply"] = lambda **kw: write_reply(state, deadline=deadline, **kw)

    decision = conversation_handlers.respond(state, body.message)

    with store.lock:
        return _apply_decision(conv, decision or {}, now, body.from_role)


def _apply_decision(conv: dict, decision: dict, now: datetime, role: str) -> dict:
    effects = decision.get("effects") or {}
    ms = store.mstate(conv["merchant_id"]) if conv.get("merchant_id") else None
    sender = store.sender_state(conv, role)
    if effects.get("auto_reply"):
        sender["auto_reply_count"] += 1
    if effects.get("real_reply"):
        sender["auto_reply_count"] = 0
        if ms is not None and role == "merchant":
            ms["unanswered"] = 0
    if ms is not None:
        if effects.get("suppress_merchant_days"):
            ms["opted_out_until"] = to_iso(now + timedelta(days=effects["suppress_merchant_days"]))
        if effects.get("merchant_wait_seconds"):
            ms["wait_until"] = to_iso(now + timedelta(seconds=effects["merchant_wait_seconds"]))
    if effects.get("customer_opt_out") and conv.get("customer_id"):
        store.cstate(conv["customer_id"])["opted_out"] = True
    conv.update(effects.get("conv_updates") or {})

    action = decision.get("action")
    rationale = decision.get("rationale") or ""
    if action == "send":
        text = (decision.get("body") or "").strip()
        if text and text not in conv["sent_bodies"]:
            conv["turns"].append({"from": "bot", "body": text, "ts": to_iso(now)})
            conv["sent_bodies"].append(text)
            if effects.get("close_after_send"):
                conv["status"] = "ended"
            return {"action": "send", "body": text, "cta": decision.get("cta") or "open_ended", "rationale": rationale}
        action, rationale = "wait", "Next message would be empty or a verbatim repeat; backing off instead."
        decision = {"wait_seconds": 3600}
    if action == "end":
        conv["status"] = "ended"
        return {"action": "end", "rationale": rationale or "Closing the conversation."}
    seconds = int(decision.get("wait_seconds") or 1800)
    conv["status"] = "waiting"
    conv["wait_until"] = to_iso(now + timedelta(seconds=seconds))
    return {"action": "wait", "wait_seconds": seconds, "rationale": rationale or "Backing off before the next nudge."}


# ---------------------------------------------------------------------------
# HTTP app
# ---------------------------------------------------------------------------

app = FastAPI(title="Vera bot", version=BOT_VERSION)


@app.get("/v1/healthz")
def healthz():
    return {"status": "ok", "uptime_seconds": int(time.time() - STARTED), "contexts_loaded": store.counts()}


@app.get("/v1/metadata")
def metadata():
    members = [m.strip() for m in os.environ.get("TEAM_MEMBERS", "").split(",") if m.strip()]
    return {
        "team_name": os.environ.get("TEAM_NAME", "TBD"),
        "team_members": members,
        "model": os.environ.get("LLM_MODEL", "gpt-5.6-luna"),
        "approach": "Rule-first router (tick policy, auto-reply/intent/opt-out handling) with an LLM composer over "
                    "trimmed 4-context fact sheets, validated and cached; deterministic template fallback.",
        "contact_email": os.environ.get("CONTACT_EMAIL", ""),
        "version": BOT_VERSION,
        "submitted_at": os.environ.get("SUBMITTED_AT", ""),
    }


@app.post("/v1/context")
async def push_context(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"accepted": False, "reason": "malformed", "details": "invalid JSON"})
    status, content = store.push(body)
    return JSONResponse(status_code=status, content=content)


@app.post("/v1/tick")
def tick(body: TickBody):
    return {"actions": run_tick(body.now, body.available_triggers)}


@app.post("/v1/reply")
def reply(body: ReplyBody):
    return handle_reply(body)


@app.post("/v1/teardown")
def teardown():
    store.reset()
    clear_runtime_cache()
    return {"wiped": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")), workers=1)
