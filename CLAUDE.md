# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

The magicpin AI Challenge: build a bot that does the job of "Vera", magicpin's WhatsApp assistant for merchants. The deliverable is an HTTP server that the judge harness drives. The repo holds the spec, a seed dataset, a local judge, and our bot:
- `bot.py`: the context store, tick policy, reply plumbing, and fallback templates
- `conversation_handlers.py`: reply decisions

The spec is split across these documents. Read them before designing anything:
- [challenge-brief.md](challenge-brief.md): what to build, including the 4-context `compose()` contract, the 5-dimension rubric (§8), the compulsion levers (§10) and the anti-patterns the judge penalizes (§11).
- [challenge-testing-brief.md](challenge-testing-brief.md): how the bot is tested, including the 5 HTTP endpoints, payload schemas, harness phases and penalties. §7 has a FastAPI skeleton to start from.
- [examples/api-call-examples.md](examples/api-call-examples.md): exact request/response pairs for every endpoint and replay scenario.
- [examples/case-studies.md](examples/case-studies.md): 10 scored "good message" anchors, two per category.
- `engagement-design.md` and `engagement-research.md` are background on magicpin's internal system. The code paths they cite (`agents/vera/followup/`, `merchant_agent.py`, vera-mcp) are **not** in this repo.

## Commands

```bash
# Setup (Windows venv). Secrets go in .env (git-ignored); copy from .env.example. requirements-dev adds httpx for TestClient.
python -m venv .venv && .venv/Scripts/python -m pip install -r requirements-dev.txt

# Expand the seeds into the full 50 merchants / 200 customers / 100 triggers, plus test_pairs.json (30 canonical pairs).
# Deterministic (SEED=20260426). --seed-dir defaults to ".", so run it from dataset/. Output is git-ignored.
cd dataset && python generate_dataset.py --out ./expanded

# Run the bot. Must be a single worker: all state is in memory.
.venv/Scripts/python -m uvicorn bot:app --host 127.0.0.1 --port 8080

# In-process contract checks (no server, no LLM). --preview writes all 30 test-pair messages to .cache/preview.md
.venv/Scripts/python tools/smoke.py --preview

# Scripted multi-turn conversations: replay scenarios, API examples 2.4-2.7, Hinglish, customer booking, restraint (-v prints turns)
.venv/Scripts/python tools/replay_tests.py -v

# LLM composer on test pairs: default 8-pair tuning set, --pairs T21,T28, --all, --mock (no API calls),
# --facts T28 (print the fact sheet). Cached prompts are free; prints run cost + total ledger spend.
.venv/Scripts/python tools/compose_pairs.py --all

# Live multi-turn conversations with LLM-written replies: transcripts, [llm]/[rule] per reply, rejections, cost (--mock = cache only)
.venv/Scripts/python tools/reply_eval.py

# Submission: writes submission.jsonl (30 lines, brief §7.2) from cached compositions and lints it (--mock = no API calls)
.venv/Scripts/python tools/make_submission.py

# Score submission.jsonl with the local judge's rubric via the bot's cached client (re-scoring unchanged lines is free).
# --pairs T01,T21 to re-score a subset; reasons/hints in .cache/scores.md
.venv/Scripts/python tools/score_submission.py

# Simulated 60-min test window: warmup, 12 ticks, mid-test injections, scripted personas; checks latency, restraint,
# repeats, opt-outs and adaptation to new context (--mock = no API calls; a live run costs about $0.01-0.02)
.venv/Scripts/python tools/sim_window.py

# Push the dataset to a running bot, like the judge's warmup (--triggers also pushes the 100 triggers)
.venv/Scripts/python tools/load_dataset.py --triggers

# Local judge. Reads BOT_URL / LLM_PROVIDER / LLM_API_KEY (falls back to OPENAI_API_KEY) / LLM_MODEL from .env.
# It re-pushes version 1 contexts, so restart the bot or POST /v1/teardown between runs (otherwise you get 409s).
.venv/Scripts/python judge_simulator.py [scenario]

# Probe the LLM key and which params the model accepts (costs < $0.01)
.venv/Scripts/python tools/llm_smoke.py
```

Judge scenarios:
- `warmup`
- `phase2_short`: scores the first 3 triggers with the LLM.
- `auto_reply_hell`
- `intent_transition`
- `hostile`
- `all`: runs warmup plus the 3 replay checks. It does **no** LLM scoring of compositions.
- `full_evaluation`: ticks every seed trigger and scores each one.

The judge needs an LLM key for every scenario, except with `ollama`.

## Project conventions (agreed)

- **Files:** the brief asks for "a single Python module". So `bot.py` is self-contained (store, composer, LLM client, templates), with the optional `conversation_handlers.py` alongside. Dev-only scripts live in `tools/` and are not submitted.
- **LLM budget:** $5 of OpenAI credit on `gpt-5.6-luna`. Rules run before the LLM; LLM calls go through a disk cache (`.cache/`) and a `MOCK_LLM` mode, and every run logs its spend. Never add an LLM call where a rule will do.
- **Auto-reply sequence** (per replay example 4.1), tracked per merchant, not per conversation:
  1. First auto-reply: send one message flagging it for the owner.
  2. Second: wait 86400s.
  3. Third: end.
- **Consent:** a customer trigger whose kind isn't in the customer's `consent.scope` is still composed for the submission, but framed as an offer. In live ticks, skip customers with no consent at all.
- **Language:**
  - **Customers** follow `language_pref`: `hi` is Hindi-first Hinglish, `hi-en mix` is Hinglish, and everything else (including te/ta/kn-en) is warm English.
  - **Owners** get English, unless their own messages in `conversation_history` are Hinglish. Forced code-mixing read badly in testing.

## Architecture the bot must implement

**Stateful HTTP server with 5 endpoints:**
- `/v1/context`, `/v1/tick`, `/v1/reply`: POST.
- `/v1/healthz`, `/v1/metadata`: GET.
- `POST /v1/teardown` is optional; on it, wipe state.

Context arrives as `(scope, context_id, version)` pushes, where scope is one of category / merchant / customer / trigger.
- Re-pushing the same or a lower version returns 409 `stale_version`.
- A higher version replaces the stored one atomically.
- Later compositions must use the newest version. The judge injects new digest items, perf snapshots and triggers mid-test and scores whether the bot adapts.

**Composition = `compose(category, merchant, trigger, customer?)`.** Resolve these links between contexts:
- `merchant.category_slug` → the category context.
- `trigger.merchant_id` / `trigger.customer_id` → the merchant and customer contexts.
- `trigger.payload.top_item_id` → an item in `category.digest[]`. Triggers carry only the ID, not the research text.

`send_as` is `"vera"` for merchant-facing triggers and `"merchant_on_behalf"` for customer-scoped triggers such as `recall_due`, `chronic_refill_due` and `appointment_tomorrow`.

**`/v1/tick` output:** zero or more actions. Every action needs:
- `conversation_id`
- `merchant_id`
- `customer_id`
- `send_as`
- `trigger_id`
- `template_name`
- `template_params`
- `body`
- `cta`
- `suppression_key`
- `rationale`

A missing field is scored 0 with a -2 penalty. The first outbound message must be framed as a WhatsApp template with `{{n}}` params. Returning an empty `actions` list is allowed, and restraint is rewarded.

**How `compose()` works (bot.py):**
1. `compose_template()` builds a fact-checked template message. It is both the fallback and the LLM's `reference_draft`.
2. `build_facts()` turns the 4 contexts into a compact fact sheet with no jargon (about 700 tokens):
   - percentages are pre-converted, signals are in plain English, and event dates carry the weekday
   - related digest items and seasonal beats are matched by whole-word keywords and event months
   - customer fact sheets exclude owner-only data and the owner's name, because it made the model address the owner
3. `llm_json()` sends the fact sheet with `SYSTEM_PROMPT` and returns JSON `{opening, detail, ask, cta, rationale}`. The body is `opening detail ask` (template `vera_compose_v1` / `merchant_compose_v1`).
4. `validate_llm()` rejects a draft that:
   - uses a number not in the facts (numbers ≤ 10 are allowed)
   - uses jargon or snake_case, contains a URL or a taboo word, or has too many emojis
   - doesn't start with the required opening, or a customer message doesn't name the business
   - mentions an offer when the business has none
   - mentions missing details
   - doesn't match the language requirement
   - is too close to a past message or a case study
5. On rejection there is one retry, carrying the problem list. If that also fails, the template is used.

**LLM plumbing:**
- Every response is cached in `.cache/llm/<sha256 of the messages>.json`. The cache is what makes `compose()` deterministic, because the model accepts no temperature 0.
- Spend is logged to `.cache/llm_ledger.jsonl`. Calls stop once the total reaches `VERA_SPEND_CAP_USD` (default 4.50).
- `MOCK_LLM=1` blocks API calls but still serves cached results. `tools/smoke.py` and `tools/replay_tests.py` force it.
- `/v1/tick` composes in parallel on a thread pool (`VERA_LLM_WORKERS`, default 8) against a deadline (`VERA_TICK_BUDGET_S`, default 9). Anything unfinished falls back to templates. Per-call timeout is `VERA_LLM_TIMEOUT_S` (default 8).

**How a reply is processed:** `bot.handle_reply` records the turn and updates per-sender repeat counters under the lock. It then calls `conversation_handlers.respond(state, message)` outside the lock, and applies the returned decision under the lock again.
- The decision can carry `effects`, which bot.py applies:
  - `auto_reply` / `real_reply`
  - `suppress_merchant_days`, `merchant_wait_seconds`
  - `customer_opt_out`
  - `conv_updates`
  - `close_after_send`
- Classifiers run in priority order: opt-out → hostile → auto-reply → commit → off-topic → defer → decline → thanks → question → default.
- The tick policy stops initiating to a merchant after 3 unanswered conversations. A real reply resets the count.
- **LLM-written replies:** the rules choose the action. For four intents (`action`, `question`, `engaged`, `customer_question`), `Kit.upgrade()` then calls the `state["write_reply"]` hook, which is `bot.write_reply`.
  - It sends the fact sheet, the last 8 turns and a TASK to `REPLY_SYSTEM_PROMPT`.
  - `validate_reply()` checks the result: numbers must come from the facts or the conversation; no re-introduction, attachments or jargon; no qualifying questions after a yes; customer replies use "we"/"hum", never "I"/"main"; the language must match; no repeats.
  - One retry, then the rule text is kept. Everything runs against a `VERA_REPLY_BUDGET_S` deadline (default 8).
  - Opt-out, hostile, auto-reply, defer, decline, thanks, off-topic, slot booking and completion replies stay rule-written.
- `VERA_DISABLE_LLM=1` skips the LLM *and* the cache. `tools/replay_tests.py` sets it so it tests only the deterministic rule layer.

**`/v1/reply` output:** `{"action": "send" | "wait" | "end", ...}`. The bot must handle:
- auto-reply detection
- intent transition ("let's do it" means act, not ask another qualifying question)
- hostile or off-topic merchants
- per-turn language switching
- never repeating a body verbatim within a conversation (-2 each time)

## Gotchas found in the provided files

- **Real field names differ from the briefs.** Category voice taboos are in `voice.vocab_taboo`, not `taboos`. Voice also has `register`, `code_mix`, `salutation_examples` and `tone_examples`. Merchants have `identity.owner_first_name` and `review_themes`. Check `dataset/*.json` rather than trusting the brief's schema snippets.
- **Generated data is sparse.** The 75 generated triggers have `payload: {"placeholder": true, "metric_or_topic": <kind>}`. The 40 generated merchants have empty `offers`, `signals` and `conversation_history`. Compose from what exists and never invent specifics. Fabrication is the heaviest penalty.
- **The seeds use trigger kinds the briefs don't list**, for example `ipl_match_today`, `cde_opportunity`, `gbp_unverified`, `supply_alert`, `winback_eligible` and `wedding_package_followup`. Routing needs a sensible default for unknown kinds.
- **The judge reads only seed files** (`dataset/categories/*.json`, `dataset/*_seed.json`), not the expanded output. `warmup` pushes only the first 5 merchants.
- **`auto_reply_hell` uses a new, never-ticked `conversation_id` on each turn** (`conv_auto_1` to `conv_auto_4`), always with the same canned text for the same merchant. `/v1/reply` must accept unknown conversation IDs, and repeat detection can't be keyed on conversation alone.
- **The local judge uses keyword heuristics.** `intent_transition` passes if the reply contains an action word (done, sending, draft, here, confirm, proceed, next) and no qualifying phrase (would you, do you, can you tell, what if, how about). `hostile` passes on `action: end`, or on a send containing sorry, apolog or won't.
- **The docs conflict on latency.** The briefs say 30s. The api-examples table says 10s for tick/reply and 5s for context. The local judge times out at 15s for tick/reply. Design for under 10s.
- **The docs conflict on URLs in the body.** The main brief allows URLs; api-call-examples F.4 says -3 per URL because Meta would reject the message. Leave URLs out of message bodies.
- **Triggers point at digest items under three different keys:** `top_item_id` (research, regulation), `digest_item_id` (`cde_opportunity`) and `alert_id` (`supply_alert`). A trigger's payload can be more specific than its digest item; for example, the batch numbers are in the `supply_alert` payload.
- **`/v1/reply` bodies vary.** Some omit `merchant_id`, so resolve it from `conversation_id`. `from_role` can be `customer` when the bot sent `merchant_on_behalf`.
- **Re-pushing the same context version returns 409**, per api-call-examples 1.5, even though the testing brief calls it a "no-op".
- **The local judge sends real wall-clock time as `now`,** so most seed triggers look expired. The tick therefore trusts the judge's `available_triggers` (and still ranks earlier expiries first). `VERA_STRICT_EXPIRY=1` enforces `expires_at`.
- **Other bot settings (env vars):**
  - `VERA_COOLDOWN_MIN` (default 30): minutes before a merchant gets another non-urgent message, so about 2 per hour at most. Urgency ≥ 4 skips the cooldown.
  - `TEAM_NAME`, `TEAM_MEMBERS` (comma-separated), `CONTACT_EMAIL`, `SUBMITTED_AT`: fill in `/v1/metadata`.
- **On Windows, use `127.0.0.1`, not `localhost`.** Each Python request to `localhost` spends about 2s trying IPv6 first, which blows the judge's latency budgets.
- **The local judge never pushes customer contexts.** Customer-scoped triggers arrive without their customer, so skip them or degrade gracefully. The official warmup does push all 200 customers.
- **The local judge was patched for fidelity.** Its scoring prompt used to show only views, calls, CTR and the raw payload, so it flagged real digest and performance numbers as "fabricated". `LLMScorer.score` now also shows:
  - the referenced digest item and the other category knowledge
  - the full performance block and peer stats
  - seasonal beats and the offer catalog
  - the customer aggregate, review themes and recent conversation
  - the full customer context

  The official judge sees the whole dataset.
- **Baseline local-judge scores** (2026-09-27, strict rubric where 7+ is good): about 39.6/50 on average across the 30 pairs.
  - Seed triggers score 40–47; placeholder triggers score 28–38, capped by missing trigger data.
  - Trigger/decision quality is the weakest dimension.
  - Judge scores vary from run to run (the model's temperature is fixed at 1), so compare subsets, not single points.
- **Dates in the dataset are internally inconsistent** (e.g. Priya's `last_visit` is later than the seed "now"). Quote payload values such as `days_until` and slot `label`s rather than doing date arithmetic.
- **Of the 30 test pairs, 13 have placeholder triggers and 9 are customer-facing.** Six (T03, T04, T08, T14, T15, T29) send reminder-type triggers to customers who only consented to `promotional_offers`.
- **Watch the customer's channel.** `whatsapp_via_parent` and `whatsapp_via_son` mean the message must address the parent or son. Names can look like `"Aanya (parent: Sneha)"`. `c_015` has no consent and isn't opted in, so never message them.
- **Extra scoring penalties:**
  - internal jargon in the body, such as signal names like `ctr_below_peer_median` (−1)
  - research or compliance claims without a source (dimension capped at 7)
  - any fabrication or repetition (every dimension capped at 5)
  - near-copies of case-study wording (plagiarism check)
  - a rationale that doesn't match the message
- **`gpt-5.6-luna` API facts** (probed 2026-09-27):
  - It rejects `max_tokens`; use `max_completion_tokens`.
  - It rejects any `temperature` except the default of 1.
  - `reasoning_effort` accepts `none` or `low` (not `minimal`). With `none`, reasoning tokens are 0.
  - JSON mode (`response_format={"type": "json_object"}`) and `seed` work.
  - A realistic compose call (~870 tokens in, ~200 out) takes about 3s and costs about $0.0004.
- **Luna is not deterministic, even with `seed`.** The same prompt gave different wording on each run. The brief requires deterministic `compose()`, so determinism must come from the disk cache keyed on a hash of the inputs, not from the model.
- **Windows console encoding:** when piping Python output on Windows (for example through the Bash tool), set `PYTHONIOENCODING=utf-8`. Otherwise the judge's `█` score bars crash, and em-dashes print as `�`. Also, `load_dotenv()` can't find `.env` from stdin scripts; pass `load_dotenv(".env")`.
- **Submission artifacts** (brief §7): `bot.py`, `submission.jsonl` (30 lines keyed by the `test_id`s in `test_pairs.json`), a `README.md` of 1 page max, and optionally `conversation_handlers.py`. Compositions must be deterministic; see the cache note above.
