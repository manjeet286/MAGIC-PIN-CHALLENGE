"""One-off probe: verifies the API key and which Chat Completions params the model accepts.

Each probe is a tiny request; the whole run costs well under $0.01.
Usage: .venv/Scripts/python tools/llm_smoke.py
"""
import os
import sys
import time

from dotenv import load_dotenv
from openai import BadRequestError, OpenAI

load_dotenv()
KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL = os.environ.get("LLM_MODEL", "gpt-5.6-luna")
PRICE_IN, PRICE_OUT = 0.20, 1.20  # $ per 1M tokens (gpt-5.6-luna, Sept 2026)

if not KEY:
    sys.exit("OPENAI_API_KEY is empty. Add it to .env first.")

client = OpenAI(api_key=KEY, timeout=30)
total_cost = 0.0


def probe(label, **extra):
    global total_cost
    msgs = [{"role": "user", "content": 'Reply with the JSON {"ok": true} and nothing else.'}]
    t0 = time.time()
    try:
        r = client.chat.completions.create(model=MODEL, messages=msgs, **extra)
    except BadRequestError as e:
        print(f"[REJECTED] {label}: {e.message[:160]}")
        return None
    dt = time.time() - t0
    u = r.usage
    reasoning = getattr(getattr(u, "completion_tokens_details", None), "reasoning_tokens", 0) or 0
    cost = u.prompt_tokens / 1e6 * PRICE_IN + u.completion_tokens / 1e6 * PRICE_OUT
    total_cost += cost
    print(f"[OK] {label}: {dt:.1f}s  in={u.prompt_tokens} out={u.completion_tokens} "
          f"(reasoning={reasoning})  ${cost:.6f}  -> {r.choices[0].message.content!r}")
    return r


print(f"Model: {MODEL}\n")
probe("baseline (no extra params)")
probe("max_completion_tokens=200", max_completion_tokens=200)
probe("max_tokens=200", max_tokens=200)
probe("temperature=0", temperature=0)
probe("json_object mode", response_format={"type": "json_object"})
for effort in ("none", "minimal", "low"):
    probe(f"reasoning_effort={effort}", reasoning_effort=effort)
probe("seed=7", seed=7)

print(f"\nTotal probe cost: ${total_cost:.6f}")
