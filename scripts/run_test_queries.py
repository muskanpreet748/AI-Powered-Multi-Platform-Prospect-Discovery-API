"""
Run the 5 practical test queries against a running API and save the responses.

Usage (with the server running on port 4000):
    python scripts/run_test_queries.py
    python scripts/run_test_queries.py --base-url https://your-app.onrender.com

Responses are written to samples/<query>.json and a summary is printed.
"""

import argparse
import json
import re
import time
from pathlib import Path

import httpx

TEST_QUERIES = [
    "Football lovers in Washington DC in USA",
    "Digital marketing agency owners in New York USA",
    "Startup founders in London UK",
    "Cricket lovers in Ahmedabad India",
    "Restaurant owners in Dubai UAE",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:4000")
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent.parent / "samples"
    out_dir.mkdir(exist_ok=True)

    with httpx.Client(base_url=args.base_url, timeout=180) as client:
        for query in TEST_QUERIES:
            started = time.perf_counter()
            response = client.post("/prospect-discovery", json={"query": query})
            elapsed = time.perf_counter() - started
            data = response.json()

            slug = re.sub(r"[^a-z0-9]+", "_", query.lower()).strip("_")
            (out_dir / f"{slug}.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

            if response.status_code != 200:
                print(f"[{response.status_code}] {query}: {data.get('error')}")
                continue
            platform_summary = ", ".join(f"{p['platform']}={p['status']}({p['profiles_found']})"
                                         for p in data["platforms"])
            print(f"[200] {query}: {data['total_prospects']} prospects in {elapsed:.1f}s "
                  f"(understanding: {data['understanding_source']})")
            print(f"      platforms: {platform_summary}")


if __name__ == "__main__":
    main()
