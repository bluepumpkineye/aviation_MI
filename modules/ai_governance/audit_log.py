"""
Synthetic audit log used by the governance section until module-level
actions write to a persistent store.
"""

import pandas as pd


def get_audit_log() -> pd.DataFrame:
    """Return a small recent action log for governance monitoring."""
    records = [
        {
            "timestamp": "2026-04-19 10:18:00",
            "module": "Demand Intelligence",
            "action": "Route forecast generated",
            "actor": "Local analyst",
            "details": "30-day forecast run for HKG-LHR",
        },
        {
            "timestamp": "2026-04-19 10:25:00",
            "module": "Campaign Studio",
            "action": "Copy variants generated",
            "actor": "Local analyst",
            "details": "Business-class copy for HKG-NRT",
        },
        {
            "timestamp": "2026-04-19 10:31:00",
            "module": "Data Layer",
            "action": "Synthetic datasets refreshed",
            "actor": "System",
            "details": "Customers, campaigns, route demand, digital touchpoints",
        },
    ]
    return pd.DataFrame(records)
