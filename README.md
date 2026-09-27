# Vera: magicpin AI Challenge submission

**Files:** `bot.py` (HTTP bot and `compose()`), `conversation_handlers.py` (`respond()`), `submission.jsonl`, and `submission_cache/` (read-only snapshot for determinism). Model: `gpt-5.6-luna`.

## Approach

**Rules decide, the LLM writes.**

Plain code handles everything a rule can decide reliably:

- **Context store:** versioned, returns 409 on stale versions.
- **Tick policy:**
  - dedupes on suppression keys and ranks by urgency
  - sends at most one message per merchant per tick, with a 30-minute cooldown (urgent triggers skip it)
  - stops after 3 unanswered conversations and never messages customers without consent
- **Reply routing**, in priority order: opt-out → hostile → auto-reply → commitment → off-topic → defer → question.
  - Auto-replies are detected per merchant: flag once, wait 24h, end.
  - "Let's do it" switches straight to action.

**Composing a message:**

1. A template writes a fact-checked draft. It is both the fallback and the reference.
2. The contexts become a compact fact sheet:
   - the digest item resolved (it may be referenced as `top_item_id`, `digest_item_id` or `alert_id`)
   - related knowledge matched by keyword and month, for judgment calls such as "weekend IPL: push delivery, not dine-in"
   - signals in plain English, weekdays on event dates
3. The LLM writes `{opening, detail, ask}`.
4. A validator rejects numbers not in the facts, taboo words, jargon, URLs, the wrong addressee or language, invented offers, and repeats. A rejected draft gets one retry, then falls back to the template.

**Replies:** questions and commitments get LLM-written text under the same checks. After a "yes", the reply contains the actual draft.

**Operations:** parallel composition with a 9s tick deadline and an 8s reply deadline, with template fallback. In production the cache is memory-only and `/v1/teardown` wipes it, so no context persists after the test.

## Trade-offs

- **No fabrication over specificity.** 13 of the 30 test triggers are detail-less placeholders. We anchor those on the merchant's real numbers instead of inventing specifics.
- **Determinism via cache.** The model rejects `temperature=0`, so each response is cached under a hash of its exact inputs. `submission_cache/` reproduces `submission.jsonl` byte-for-byte.
- **Language.** Customers get their `language_pref`. Owners get English unless they write in Hinglish themselves, because forced code-mixing read badly.
- **A small, fast model** (about 2–3s and $0.0005 per message) with strong guardrails, and restraint over volume (about 2 non-urgent messages per merchant per hour at most).

**Local check:** we used the provided judge, patched to see full context and run on the same model, so these numbers are indicative only. Our 30 messages averaged about 39.5/50. On the same 10 triggers, the brief's case studies scored 38.9 and ours 42.0.

## What context would have helped most

1. **Details for placeholder triggers:** appointment times, refill molecules, the actual milestone.
2. **Merchant calendars and open slots**, plus delivery or pickup capability.
3. **Outcomes of past messages** (replied, booked, ignored).
4. **An explicit language preference for owners.**

**Run:** `pip install -r requirements.txt`, then `uvicorn bot:app --port 8080 --workers 1`. In production set `OPENAI_API_KEY`, `VERA_CACHE_MODE=memory` and `VERA_SPEND_CAP_USD`.
