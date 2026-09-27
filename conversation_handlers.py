"""Reply handling for Vera conversations (brief §7.4).

respond(state, merchant_message) -> {"action": "send" | "wait" | "end", ...}

`state` is assembled by bot.handle_reply: conversation (turns so far), merchant, category, customer,
trigger, merchant_state, sender_state (per-sender counters incl. inbound_repeat / auto_reply_count),
from_role, turn_number, now. The returned dict may carry an "effects" map that bot.py applies:
    auto_reply / real_reply            bump or reset the sender's auto-reply counter
    suppress_merchant_days             stop initiating to this merchant for N days
    merchant_wait_seconds              pause new sends to this merchant
    customer_opt_out                   never message this customer again
    conv_updates                       fields merged into the conversation (e.g. lang, mode)
    close_after_send                   close the conversation once this message is sent
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Language
# ---------------------------------------------------------------------------

HINGLISH_MARKERS = {
    "hai", "hain", "nahi", "nahin", "kya", "karo", "karna", "kar", "chahiye", "mujhe", "hume", "humein", "aap", "aapka",
    "aapke", "aapki", "haan", "theek", "thik", "bhai", "abhi", "baad", "kal", "mein", "hoon", "kaise", "kitna",
    "acha", "accha", "achha", "chalega", "bolo", "batao", "dijiye", "karein", "karenge", "wala", "wali", "bahut",
    "shukriya", "dhanyavaad", "matlab", "lekin", "toh", "yeh", "woh", "kyun", "kyu", "hota", "hogi", "hoga", "raha",
    "rahi", "sakte", "sakti", "jaldi", "zaroor", "bilkul", "koi", "kuch", "sab", "mera", "meri", "hamara", "hamari",
}


def detect_lang(text: str, fallback: str = "en") -> str:
    if re.search(r"[ऀ-ॿ]", text):
        return "hi"
    tokens = re.findall(r"[a-z]+", text.lower())
    hits = sum(t in HINGLISH_MARKERS for t in tokens)
    if hits >= 2 or (hits and hits / len(tokens) >= 0.34) or (hits and fallback == "hi"):
        return "hi"
    return fallback if len(tokens) <= 2 else "en"


# ---------------------------------------------------------------------------
# Classifiers (English + Hinglish + Devanagari)
# ---------------------------------------------------------------------------

OPT_OUT = re.compile(
    r"\b(stop|unsubscribe|opt[ -]?out|remove me|don'?t (message|text|contact|send|msg)|do not (message|text|contact|send)"
    r"|not interested|no longer interested|never message|band karo|bandh karo|mat bhejo|mat bhejiye|message mat|msg mat"
    r"|nahi chahiye|nahin chahiye|interest nahi|interested nahi)\b", re.I)
OPT_OUT_DEV = re.compile(r"बंद कर|मत भेज|नहीं चाहिए|रुचि नहीं")
HOSTILE = re.compile(
    r"\b(useless|spam\w*|bother\w*|irritat\w*|annoy\w*|harass\w*|nonsense|bakwas|bekaar|bekar|pagal|idiot|stupid"
    r"|shut up|get lost|chup|faltu|fraud|scam\w*|bloody|damn|wtf|f+u+c+k\w*|bullshit|go to hell|pareshan)\b"
    r"|why (are|r) (you|u) (messaging|texting|bothering|sending)", re.I)
AUTO_REPLY = re.compile(
    r"thank(s| you) for (contacting|reaching out|reaching|your message|messaging|writing)"
    r"|(we|our team|team) will (get back|respond|revert|reply|contact|call)"
    r"|get back to you (shortly|soon|as soon)|will (respond|reply|revert) (shortly|soon|at the earliest)"
    r"|(currently|presently) (unavailable|away|closed|out of (the )?office)|out of office"
    r"|auto(matic|mated)?[- ]?(reply|response|message)|this is an automated|i am an automated|i'?m an automated"
    r"|automated assistant|(our )?(business|working|office) hours|we are closed"
    r"|for (urgent|immediate) (queries|assistance|help)|please (leave|drop) (a|your) (message|query)"
    r"|jaankari ke liye .*shukriya|team tak pahuncha|sampark karne ke liye (dhanyavaad|shukriya)|jald hi (sampark|jawab)"
    r"|संपर्क करने के लिए धन्यवाद|जल्द ही (संपर्क|जवाब)|स्वचालित", re.I)
INFO_REQUEST = re.compile(r"\b(tell me more|more (info|information|details)|explain|details (bhejo|do|dijiye)|batao|bataiye|samjhao)\b", re.I)
QUESTION_START = re.compile(r"^\s*(what|how|when|where|why|which|who|can|could|is|are|does|do|kya|kaise|kitna|kitne|kab|kaun|kahan|kyun)\b", re.I)
COMMIT_STRONG = re.compile(
    r"\b(let'?s do (it|this)|lets do (it|this)|go ahead|do it|go for it|sign me up|(i )?want to join|i'?m in|count me in"
    r"|please do|yes please|haan karo|kar do|kardo|shuru karo|chalo karte|judna hai|jodna hai|join karna"
    r"|proceed|send it|send (me )?(the|it)|book it|i agree|approved?|confirm(ed)?|go live|publish it)\b", re.I)
COMMIT_START = re.compile(r"^\s*(yes|yeah|yep|yup|sure|haan|han|ji haan|ok(ay)?|theek hai|thik hai|chalega|done|great|perfect)\b", re.I)
DECLINE = re.compile(r"^\s*(no|nope|nah|nahi|nahin|na|no thanks|no thank you|not really|not needed|zaroorat nahi)\b", re.I)
DEFER = re.compile(
    r"\b(busy|later|baad mein|baad me|abhi nahi|not now|call (me )?later|in a meeting|meeting mein|kal|tomorrow"
    r"|next week|agle hafte|shaam ko|this evening|tonight|will check|dekhta hoon|dekhti hoon|dekh ke batata|remind me"
    r"|give me (some )?time|thoda time)\b", re.I)
THANKS = re.compile(r"^\s*(thanks|thank you|thx|ty|shukriya|dhanyavaad|dhanyawad|noted|ok thanks|okay thanks|great thanks|🙏|👍)[\s!.🙏👍]*$", re.I)
SLOT_PICK = re.compile(r"^\s*(?:option\s*)?(1|2|3|one|two|three|first|second|third|pehla|pehli|doosra|dusra|doosri|dusri)\b", re.I)
SLOT_INDEX = {"1": 0, "one": 0, "first": 0, "pehla": 0, "pehli": 0, "2": 1, "two": 1, "second": 1, "doosra": 1,
              "dusra": 1, "doosri": 1, "dusri": 1, "3": 2, "three": 2, "third": 2}
OFF_TOPIC = [
    (re.compile(r"\bgst\b", re.I), "GST filing", "your CA"),
    (re.compile(r"\b(itr|income[- ]?tax|tax (filing|return))\b", re.I), "income-tax filing", "your CA"),
    (re.compile(r"\b(loan|emi|credit card)\b", re.I), "loans and credit", "your bank"),
    (re.compile(r"\binsurance\b", re.I), "insurance", "your insurance advisor"),
    (re.compile(r"\b(visa|passport)\b", re.I), "visa and passport work", "an authorised agent"),
    (re.compile(r"\b(electricity|light) bill|recharge|bank account\b", re.I), "bills and banking", "your bank or provider"),
    (re.compile(r"\b(share|stock) market|mutual fund|crypto\b", re.I), "investments", "a financial advisor"),
]

DELIVERABLE = {
    "research_digest": "the 1-page summary and a {audience} WhatsApp draft",
    "regulation_change": "a 3-point compliance checklist",
    "cde_opportunity": "the session details and a calendar reminder",
    "supply_alert": "the customer WhatsApp note and the replacement-pickup steps",
    "category_seasonal": "the shelf plan and a WhatsApp note for your regulars",
    "festival_upcoming": "the festival post draft",
    "ipl_match_today": "today's delivery push",
    "competitor_opened": "your refreshed Google listing copy",
    "perf_dip": "a short diagnosis with 2 fixes",
    "perf_spike": "a follow-up post draft",
    "seasonal_perf_dip": "the retention push for your members",
    "milestone_reached": "the thank-you post draft",
    "review_theme_emerged": "the public review reply and the fix announcement",
    "dormant_with_vera": "your 2-minute performance snapshot",
    "curious_ask_due": "the Google post draft",
    "renewal_due": "the renewal details",
    "winback_eligible": "the plan to restart your profile updates",
    "gbp_unverified": "the verification steps",
    "active_planning_intent": "the first draft",
}
AUDIENCE = {"dentists": "patient", "pharmacies": "customer", "salons": "client", "gyms": "member", "restaurants": "customer"}


def _slot_labels(trigger: dict) -> list[str]:
    payload = trigger.get("payload") or {}
    opts = payload.get("available_slots") or payload.get("next_session_options") or []
    return [o.get("label") for o in opts if isinstance(o, dict) and o.get("label")]


class Kit:
    """Everything a reply template needs, in the conversation's language."""

    def __init__(self, state: dict, lang: str):
        self.lang = lang
        self.conv = state.get("conversation") or {}
        self.trigger = state.get("trigger") or {}
        self.merchant = state.get("merchant") or {}
        self.category = state.get("category") or {}
        self.customer = state.get("customer") or {}
        self.kind = self.conv.get("kind") or self.trigger.get("kind") or ""
        self.bot_turns = sum(1 for t in self.conv.get("turns") or [] if t.get("from") == "bot")
        self.writer = state.get("write_reply")

    def upgrade(self, decision: dict, intent: str) -> dict:
        """Let the LLM rewrite a rule-chosen send (same action, same effects); keep the rule text on any failure."""
        if not self.writer or decision.get("action") != "send":
            return decision
        try:
            out = self.writer(intent=intent, reference=decision["body"], cta=decision["cta"], lang=self.lang)
        except Exception:
            out = None
        if not out:
            return decision
        return {**decision, "body": out["body"], "cta": out["cta"],
                "rationale": f"{decision['rationale']} (LLM-written: {out['rationale']})" if out.get("rationale")
                else decision["rationale"]}

    @property
    def d(self) -> str:
        audience = AUDIENCE.get(self.category.get("slug") or self.merchant.get("category_slug"), "customer")
        text = DELIVERABLE.get(self.kind, "the details").format(audience=audience)
        return re.sub(r"^the ", "", text) if self.lang == "hi" else text

    @property
    def merchant_name(self) -> str:
        return (self.merchant.get("identity") or {}).get("name") or "our team"

    def t(self, en, hi):
        """Pick the language, then a variant that differs from the previous bot turn."""
        options = hi if self.lang == "hi" else en
        options = options if isinstance(options, list) else [options]
        return options[self.bot_turns % len(options)]


def _send(body: str, cta: str, rationale: str, **effects) -> dict:
    return {"action": "send", "body": body, "cta": cta, "rationale": rationale, "effects": effects}


def _is_auto_reply(text: str, sender: dict) -> bool:
    if AUTO_REPLY.search(text):
        return True
    return int(sender.get("inbound_repeat") or 0) >= 2 and len(text) >= 20


def _is_commit(text: str) -> bool:
    strong = bool(COMMIT_STRONG.search(text)) and not QUESTION_START.search(text)
    return strong or (bool(COMMIT_START.search(text)) and "?" not in text)


def _off_topic(text: str):
    return next(((topic, advisor) for rx, topic, advisor in OFF_TOPIC if rx.search(text)), None)


def _profile_lang(state: dict, role: str) -> str:
    if role == "customer":
        pref = ((state.get("customer") or {}).get("identity") or {}).get("language_pref", "").lower()
        return "hi" if pref.startswith("hi") else "en"
    return "en"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def respond(state: dict, merchant_message: str) -> dict:
    conv = state.get("conversation") or {}
    role = state.get("from_role") or "merchant"
    text = (merchant_message or "").strip()
    sender = state.get("sender_state") or {}
    lang = detect_lang(text, fallback=conv.get("lang") or _profile_lang(state, role))
    k = Kit(state, lang)

    decision = _decide(text, role, sender, k)
    effects = decision.setdefault("effects", {})
    effects.setdefault("conv_updates", {})["lang"] = lang
    return decision


def _decide(text: str, role: str, sender: dict, k: Kit) -> dict:
    if not text:
        return {"action": "wait", "wait_seconds": 1800, "rationale": "Empty message; nothing to respond to yet."}

    if OPT_OUT.search(text) or OPT_OUT_DEV.search(text):
        effects = {"customer_opt_out": True} if role == "customer" else {"suppress_merchant_days": 30}
        return {"action": "end", "effects": {**effects, "real_reply": True},
                "rationale": "Explicit opt-out; closing and suppressing further outreach for 30 days."}

    if HOSTILE.search(text):
        effects = {"customer_opt_out": True} if role == "customer" else {"suppress_merchant_days": 30}
        return {"action": "end", "effects": {**effects, "real_reply": True},
                "rationale": "Merchant frustration is explicit; exiting gracefully and suppressing outreach for 30 days."}

    if _is_auto_reply(text, sender):
        n = int(sender.get("auto_reply_count") or 0) + 1
        if n == 1:
            body = k.t(f"Looks like an automatic reply. When the owner sees this, just reply YES and I'll send {k.d}.",
                       f"Lagta hai yeh auto-reply hai. Owner jab dekhein, bas YES reply kar dein, main {k.d} bhej dungi.")
            return _send(body, "binary_yes_no", "Canned auto-reply detected; one explicit prompt flagged for the owner.",
                         auto_reply=True)
        if n == 2:
            return {"action": "wait", "wait_seconds": 86400, "effects": {"auto_reply": True, "merchant_wait_seconds": 86400},
                    "rationale": "Same auto-reply again, so the owner isn't at the phone; waiting 24h before retrying."}
        return {"action": "end", "effects": {"auto_reply": True},
                "rationale": f"Auto-reply {n} times in a row with no real reply; closing to stop wasting turns."}

    if role == "customer":
        return _customer(text, k)
    return _merchant(text, k)


def _merchant(text: str, k: Kit) -> dict:
    real = {"real_reply": True}

    if _is_commit(text):
        if k.conv.get("mode") == "action":
            body = k.t(f"Done, I've finalised {k.d}. I'll report back with how it performs in 7 days.",
                       f"Done! Maine {k.d} final kar diya hai. 7 din mein results ke saath update dungi.")
            return _send(body, "none", "Merchant confirmed the delivered work; closing the loop with a follow-up promise.",
                         close_after_send=True, **real)
        body = k.t(f"On it: drafting {k.d} now, ready here in a few minutes. Nothing goes live without your OK; "
                   "reply CONFIRM once you've checked it.",
                   f"Shuru kar rahi hoon: {k.d} ready kar rahi hoon, kuch minute mein yahin bhej dungi. "
                   "Aapke OK ke bina kuch live nahi hoga; check karke CONFIRM reply kar dijiye.")
        return k.upgrade(_send(body, "binary_confirm_cancel",
                               "Explicit commitment; switching from pitch to action immediately (no further qualifying).",
                               conv_updates={"mode": "action"}, **real), "action")

    off = _off_topic(text)
    if off:
        topic, advisor = off
        body = k.t([f"{topic[:1].upper() + topic[1:]} is best handled by {advisor}; it's outside what I can help with. "
                    f"Coming back to our thread: shall I go ahead with {k.d}? Reply YES.",
                    f"Sorry, I can't help with {topic}; {advisor} is the right person for that. "
                    f"Meanwhile I can start on {k.d} whenever you reply YES."],
                   [f"{topic[:1].upper() + topic[1:]} ke liye {advisor} hi sahi rahenge; woh mere scope ke bahar hai. "
                    f"Wapas apne topic par: {k.d} shuru kar doon? YES reply kar dijiye.",
                    f"{topic[:1].upper() + topic[1:]} mein main help nahi kar paungi, {advisor} se baat kar lijiye. "
                    f"Aapke YES pe main {k.d} shuru kar dungi."])
        return _send(body, "binary_yes_no", "Out-of-scope ask declined politely; redirected to the original thread.", **real)

    if DEFER.search(text):
        low = text.lower()
        if re.search(r"next week|agle hafte", low):
            secs = 604800
        elif re.search(r"tomorrow|\bkal\b", low):
            secs = 86400
        elif re.search(r"evening|tonight|shaam", low):
            secs = 21600
        else:
            secs = 7200
        return {"action": "wait", "wait_seconds": secs, "effects": {**real, "merchant_wait_seconds": secs},
                "rationale": f"Merchant asked for time; backing off {secs // 3600}h before following up."}

    if DECLINE.search(text) and len(text.split()) <= 4:
        return {"action": "end", "effects": real, "rationale": "Merchant declined; closing politely without pushing."}

    if THANKS.search(text):
        body = k.t("Anytime! I'll check back with an update next week.",
                   "Koi baat nahi! Agle hafte update ke saath wapas aati hoon.")
        return _send(body, "none", "Merchant closed with thanks; short sign-off and closing the conversation.",
                     close_after_send=True, **real)

    if "?" in text or QUESTION_START.search(text) or INFO_REQUEST.search(text):
        body = k.t([f"Good question. Let me pull the exact details and reply here shortly. Meanwhile I can start on {k.d}; "
                    "reply YES to go ahead.",
                    f"Checking that for you now; I'll confirm the specifics in this chat. I can start on {k.d} "
                    "whenever you reply YES."],
                   [f"Achha sawaal hai. Exact details check karke yahin reply karti hoon. Tab tak {k.d} shuru kar sakti hoon; "
                    "YES reply kar dijiye.",
                    f"Yeh abhi check kar rahi hoon, details isi chat mein confirm karungi. Aapke YES pe main {k.d} shuru kar dungi."])
        return k.upgrade(_send(body, "binary_yes_no",
                               "Merchant asked a question; answered from the facts and kept a single next step.", **real),
                         "question")

    body = k.t([f"Got it, noted. I'll factor that in. Want me to go ahead with {k.d}? Reply YES.",
                f"Understood, thanks for sharing. Say YES whenever you want me to start on {k.d}."],
               [f"Samajh gayi, note kar liya. {k.d} ke saath aage badhun? YES reply kar dijiye.",
                f"Theek hai, thanks for sharing. Jab chahein YES bhej dijiye, main {k.d} shuru kar dungi."])
    return k.upgrade(_send(body, "binary_yes_no", "Engaged reply; acknowledged and kept one low-friction next step.", **real),
                     "engaged")


def _customer(text: str, k: Kit) -> dict:
    real = {"real_reply": True}
    slots = _slot_labels(k.trigger)
    payload = k.trigger.get("payload") or {}

    pick = SLOT_PICK.search(text)
    chosen = None
    if pick and slots:
        idx = SLOT_INDEX.get(pick.group(1).lower())
        chosen = slots[idx] if idx is not None and idx < len(slots) else None
    if chosen is None and slots:
        text_words = re.findall(r"[a-z]+", text.lower())
        chosen = next((s for s in slots if any(w.startswith(s.split()[0].lower()[:3]) for w in text_words)), None)
    if chosen:
        body = k.t(f"Booked: {chosen} at {k.merchant_name}. We'll send a reminder the day before. "
                   "Reply CHANGE if you need a different time.",
                   f"Booked: {chosen}, {k.merchant_name} mein. Ek din pehle reminder bhej denge. "
                   "Time badalna ho toh CHANGE reply karein.")
        return _send(body, "none", "Customer picked a slot; confirmed the booking with a reminder promise.",
                     conv_updates={"booked_slot": chosen}, **real)

    if _is_commit(text):
        if k.kind in ("recall_due", "trial_followup") and len(slots) > 1:
            options = " or ".join(f"{i}) {s}" for i, s in enumerate(slots, 1))
            body = k.t(f"Great! Which one works for you: {options}? Or tell us a time that suits you.",
                       f"Badhiya! Kaunsa slot theek rahega: {options}? Ya apna time bata dijiye.")
            return _send(body, "multi_choice_slot", "Customer said yes; offering the real open slots to complete booking.", **real)
        if slots:
            body = k.t(f"Booked: {slots[0]} at {k.merchant_name}. Reply CHANGE if you need a different time.",
                       f"Booked: {slots[0]}, {k.merchant_name} mein. Time badalna ho toh CHANGE reply karein.")
            return _send(body, "none", "Customer said yes to the only open slot; booked it.",
                         conv_updates={"booked_slot": slots[0]}, **real)
        if k.kind == "chronic_refill_due":
            mols = ", ".join(payload.get("molecule_list") or []) or "your usual medicines"
            delivery = " for home delivery" if payload.get("delivery_address_saved") else ""
            body = k.t(f"Confirmed: we'll keep the same pack ({mols}) ready{delivery}. We'll message you once it's on the way.",
                       f"Confirm ho gaya: same pack ({mols}) ready rakhenge{delivery}. Nikalte hi message kar denge.")
            return _send(body, "none", "Customer confirmed the refill; confirmed dispatch without inventing a delivery time.",
                         **real)
        if k.kind == "appointment_tomorrow":
            body = k.t("Confirmed, see you tomorrow! Reply CHANGE if anything comes up.",
                       "Confirm ho gaya, kal milte hain! Kuch badle toh CHANGE reply karein.")
            return _send(body, "none", "Customer confirmed tomorrow's appointment.", **real)
        body = k.t("Great! Reply with a day and time that suits you and we'll hold it.",
                   "Badhiya! Apna din aur time bata dijiye, hum hold kar lenge.")
        return _send(body, "open_ended", "Customer said yes; asking only for the time needed to book.", **real)

    if DEFER.search(text):
        return {"action": "wait", "wait_seconds": 86400, "effects": real,
                "rationale": "Customer asked for time; following up tomorrow."}

    if DECLINE.search(text) and len(text.split()) <= 4:
        return {"action": "end", "effects": real, "rationale": "Customer declined; closing without pushing."}

    if THANKS.search(text):
        body = k.t("Thank you! See you soon.", "Dhanyavaad! Jald milte hain.")
        return _send(body, "none", "Customer closed with thanks.", close_after_send=True, **real)

    body = k.t(["Thanks! We'll check and confirm here shortly.",
                "Noted, thank you. We'll get back to you on this in this chat."],
               ["Thanks! Check karke yahin confirm karte hain.",
                "Note kar liya, dhanyavaad. Isi chat mein jawab dete hain."])
    return k.upgrade(_send(body, "none", "Customer question or request; answered from the facts or promised a follow-up.",
                           **real), "customer_question")
