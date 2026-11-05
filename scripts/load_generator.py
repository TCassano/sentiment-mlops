"""Replay sample social posts against the API (stdlib only, runs in a bare python image).

Usage: python scripts/load_generator.py --url http://localhost:8000 --rps 2
``--drift-after N`` switches to off-domain texts after N requests, to see the
drift panels and the low-confidence alert react in Grafana.
"""

from __future__ import annotations

import argparse
import json
import random
import time
import urllib.error
import urllib.request

IN_DOMAIN = [
    "Just got the new phone from @acme and I love it! 😍",
    "Worst customer service ever, waited 2 hours on hold #fail",
    "The store on Main Street opens at 9am tomorrow",
    "Thanks @acme_support, problem solved in 5 minutes!",
    "My package arrived broken. Again. Not happy.",
    "Anyone tried the new update? Thinking about installing it",
    "Honestly the best coffee in town ☕️",
    "App keeps crashing since this morning, so annoying",
    "Watching the launch event live https://t.co/xyz",
    "Not sure how I feel about the new logo tbh",
]
OFF_DOMAIN = [
    "Pursuant to clause 4.2, the lessee shall indemnify the lessor against all claims.",
    "Il servizio clienti ha risposto dopo tre giorni, ma il problema persiste.",
    "SELECT * FROM orders WHERE status = 'pending' ORDER BY created_at DESC;",
    "Net interest margin contracted by 12 bps quarter-on-quarter amid deposit repricing.",
]


def post(url: str, text: str) -> int:
    req = urllib.request.Request(
        f"{url}/predict",
        data=json.dumps({"text": text}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status
    except urllib.error.HTTPError as exc:
        return exc.code
    except OSError:
        return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--rps", type=float, default=2.0)
    parser.add_argument("--drift-after", type=int, default=0, help="0 = never drift")
    args = parser.parse_args()

    sent = 0
    while True:
        pool = OFF_DOMAIN if args.drift_after and sent >= args.drift_after else IN_DOMAIN
        status = post(args.url, random.choice(pool))
        if status == 0:
            time.sleep(5)  # API not up yet
            continue
        sent += 1
        time.sleep(1 / args.rps)


if __name__ == "__main__":
    main()
