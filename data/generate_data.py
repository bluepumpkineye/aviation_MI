"""
Master Data Generator - run this once to create all data files.
Output lands in data/raw/ as CSV files.

Run from project root:
    python data/generate_data.py
"""

import sys
import os

# Make sure project root is on the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from pathlib import Path
from data.generators.customer_data   import generate_customers
from data.generators.campaign_data   import generate_campaigns
from data.generators.route_demand_data import generate_route_demand
from data.generators.digital_data    import generate_digital_touchpoints

OUTPUT_DIR = Path(__file__).parent / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 55)
    print("  Aviation Marketing AI - Data Generator")
    print("=" * 55)

    steps = [
        ("Customer Intelligence data", generate_customers,           "customers.csv"),
        ("Campaign performance data",  generate_campaigns,           "campaigns.csv"),
        ("Route demand data",          generate_route_demand,        "route_demand.csv"),
        ("Digital touchpoint data",    generate_digital_touchpoints, "digital_touchpoints.csv"),
    ]

    for label, fn, filename in steps:
        print(f"\n  Generating {label}...", end=" ")
        df = fn()
        path = OUTPUT_DIR / filename
        df.to_csv(path, index=False)
        print(f"Done.  {len(df):,} rows -> {filename}")

    print("\n" + "=" * 55)
    print("  All data files created in data/raw/")
    print("=" * 55)


if __name__ == "__main__":
    main()
