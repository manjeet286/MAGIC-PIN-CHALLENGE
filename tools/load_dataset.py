"""Push the dataset to a running bot, the way the judge's warmup does.

Usage:
    python tools/load_dataset.py [--url http://127.0.0.1:8080] [--source expanded|seed] [--triggers]
"""
import argparse
import json
import sys
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"


def load_dataset(source: str = "expanded") -> dict[str, list[tuple[str, dict]]]:
    """scope -> [(context_id, payload)] for categories, merchants, customers, triggers."""
    def read(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    out = {"category": [(d["slug"], d) for d in map(read, sorted((DATASET / "categories").glob("*.json")))]}
    if source == "expanded":
        base = DATASET / "expanded"
        if not base.exists():
            sys.exit("dataset/expanded missing: run `cd dataset && python generate_dataset.py --out ./expanded`")
        for scope, folder, key in (("merchant", "merchants", "merchant_id"), ("customer", "customers", "customer_id"),
                                   ("trigger", "triggers", "id")):
            out[scope] = [(d[key], d) for d in map(read, sorted((base / folder).glob("*.json")))]
    else:
        for scope, fname, container, key in (("merchant", "merchants_seed.json", "merchants", "merchant_id"),
                                             ("customer", "customers_seed.json", "customers", "customer_id"),
                                             ("trigger", "triggers_seed.json", "triggers", "id")):
            out[scope] = [(d[key], d) for d in read(DATASET / fname)[container]]
    return out


def request(url: str, method: str = "GET", body: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8") or "{}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--source", choices=["expanded", "seed"], default="expanded")
    ap.add_argument("--triggers", action="store_true", help="also push triggers (the judge pushes them during the test)")
    ap.add_argument("--version", type=int, default=1)
    args = ap.parse_args()

    data = load_dataset(args.source)
    scopes = ["category", "merchant", "customer"] + (["trigger"] if args.triggers else [])
    for scope in scopes:
        statuses = Counter()
        for cid, payload in data[scope]:
            status, _ = request(f"{args.url}/v1/context", "POST", {
                "scope": scope, "context_id": cid, "version": args.version, "payload": payload,
                "delivered_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")})
            statuses[status] += 1
        print(f"{scope:9} {len(data[scope]):4} pushed  {dict(statuses)}")
    print("healthz:", request(f"{args.url}/v1/healthz")[1])


if __name__ == "__main__":
    main()
