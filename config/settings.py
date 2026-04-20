# ─────────────────────────────────────────────────────────────────
# GLOBAL SETTINGS — Aviation Marketing AI Platform
# All business constants live here. Easy to update in one place.
# ─────────────────────────────────────────────────────────────────

# ── Routes ───────────────────────────────────────────────────────
ROUTES = [
    "HKG-LHR", "HKG-JFK", "HKG-SYD", "HKG-NRT", "HKG-SIN",
    "HKG-LAX", "HKG-YVR", "HKG-FRA", "HKG-CDG", "HKG-DXB",
    "HKG-BKK", "HKG-TPE", "HKG-ICN", "HKG-MEL", "HKG-MNL"
]

# ── Markets ───────────────────────────────────────────────────────
MARKETS = ["HK", "AU", "UK", "JP", "SG", "US", "CN", "TW", "KR", "TH"]

# ── Customer Segments ────────────────────────────────────────────
LOYALTY_TIERS   = ["Standard", "Silver", "Gold", "Diamond"]
CABIN_CLASSES   = ["Economy", "Premium Economy", "Business", "First"]
AGE_BANDS       = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
NATIONALITIES   = ["HK", "AU", "GB", "JP", "SG", "US", "CN", "TW", "KR", "TH"]

# ── Marketing Channels ───────────────────────────────────────────
CHANNELS = [
    "Paid Search",
    "Paid Social",
    "Programmatic Display",
    "Email CRM",
    "YouTube / Video",
    "Affiliate",
    "Out-of-Home",
    "Organic Search"
]

DIGITAL_CHANNELS = [
    "Paid Search",
    "Paid Social",
    "Programmatic Display",
    "Email CRM",
    "YouTube / Video",
    "Affiliate",
    "Organic Search",
]

# ── Data Volume (synthetic) ──────────────────────────────────────
N_CUSTOMERS      = 8_000
N_CAMPAIGNS      = 500
N_ROUTE_DAYS     = 730    # 2 years daily
N_TOUCHPOINTS    = 50_000

# ── Currency ─────────────────────────────────────────────────────
CURRENCY         = "HKD"
CURRENCY_SYMBOL  = "HK$"

# ── App Meta ─────────────────────────────────────────────────────
APP_TITLE        = "Aviation Marketing Intelligence Platform"
APP_SUBTITLE     = "AI/ML Decision Engine for Marketing Leadership"
APP_VERSION      = "1.0.0"
