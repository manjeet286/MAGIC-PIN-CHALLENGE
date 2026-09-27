"""Snapshot the LLM responses behind submission.jsonl into submission_cache/ (read-only, shipped with the bot),
then verify compose() reproduces submission.jsonl from that snapshot alone (memory mode, no disk cache, no API).

Usage: python tools/export_submission_cache.py     (run after tools/make_submission.py)
"""
import json
import os
import sys
from pathlib import Path

os.environ["MOCK_LLM"] = "1"  # never call the API here
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import bot  # noqa: E402
from load_dataset import load_dataset  # noqa: E402
from make_submission import KEYS  # noqa: E402


def compose_all(data, pairs) -> list[dict]:
    cats, merchants = dict(data["category"]), dict(data["merchant"])
    customers, triggers = dict(data["customer"]), dict(data["trigger"])
    out = []
    for p in pairs:
        m = merchants[p["merchant_id"]]
        msg = bot.compose(cats[m["category_slug"]], m, triggers[p["trigger_id"]], customers.get(p["customer_id"]))
        out.append({"test_id": p["test_id"], **{k: msg[k] for k in KEYS[1:]}, "_composer": msg["composer"]})
    return out


def main() -> None:
    data = load_dataset("expanded")
    pairs = json.loads((ROOT / "dataset/expanded/test_pairs.json").read_text(encoding="utf-8"))["pairs"]
    expected = [json.loads(l) for l in (ROOT / "submission.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    strip = lambda rows: [{k: v for k, v in r.items() if k != "_composer"} for r in rows]  # noqa: E731

    bot.CACHE_KEYS_USED.clear()
    current = compose_all(data, pairs)
    if strip(current) != expected:
        sys.exit("submission.jsonl doesn't match the current composer output: run tools/make_submission.py first")
    keys = sorted(bot.CACHE_KEYS_USED)

    snapshot = {}
    for key in keys:
        for src in (bot.LLM_CACHE_DIR / f"{key}.json", bot.SHIPPED_CACHE_DIR / f"{key}.json"):
            if src.exists():
                snapshot[key] = src.read_text(encoding="utf-8")
                break
    missing = [k for k in keys if k not in snapshot]
    if missing:
        sys.exit(f"{len(missing)} cache entries not found on disk; cannot snapshot")
    bot.SHIPPED_CACHE_DIR.mkdir(exist_ok=True)
    for old in bot.SHIPPED_CACHE_DIR.glob("*.json"):
        old.unlink()
    for key, text in snapshot.items():
        (bot.SHIPPED_CACHE_DIR / f"{key}.json").write_text(text, encoding="utf-8")

    bot.CACHE_MODE = "memory"  # production mode: only the shipped snapshot is visible
    bot.clear_runtime_cache()
    replay = compose_all(data, pairs)
    ok = strip(replay) == expected and all(r["_composer"] == "llm" for r in replay)
    size_kb = sum(p.stat().st_size for p in bot.SHIPPED_CACHE_DIR.glob("*.json")) / 1024
    print(f"Snapshot: {len(snapshot)} responses ({size_kb:.0f} KB) in {bot.SHIPPED_CACHE_DIR.name}/")
    print("Verified: compose() reproduces submission.jsonl from the snapshot alone" if ok
          else "MISMATCH: snapshot does not reproduce submission.jsonl")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
