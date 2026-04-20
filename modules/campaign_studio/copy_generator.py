"""
Copy Generator
Generates structured marketing copy variations for different
customer segments, routes and cabin classes.
Uses template-based generation with dynamic personalisation.
No LLM or external API required — fully offline.
"""

import random
import pandas as pd
import numpy as np
from config.settings import ROUTES, CABIN_CLASSES, MARKETS

# ── Seed for reproducibility ──────────────────────────────────────
random.seed(42)
rng = np.random.default_rng(42)

# ── Copy Component Libraries ──────────────────────────────────────

HEADLINES = {
    "Economy": [
        "The World Is Closer Than You Think",
        "Your Next Adventure Starts Here",
        "Fly Further for Less",
        "Great Destinations. Smart Fares.",
        "More World. More Value.",
        "Where Will You Go Next?",
    ],
    "Premium Economy": [
        "More Room to Breathe. More Room to Dream.",
        "The Upgrade You Deserve",
        "Comfort Redefined at 35,000 Feet",
        "Smart Travellers Choose Premium Economy",
        "Space, Comfort and Value — All in One",
        "Travel Better. Feel the Difference.",
    ],
    "Business": [
        "Arrive Ready for Anything",
        "Your Office in the Sky",
        "Business Class. Uncompromised.",
        "Work. Rest. Arrive Refreshed.",
        "The Way Business Was Meant to Travel",
        "Lead from the Front — Even at 35,000 Feet",
    ],
    "First": [
        "An Experience Beyond the Journey",
        "Luxury Has a New Altitude",
        "The Art of Travel, Perfected",
        "Where the Journey Becomes the Destination",
        "First Class in Every Sense",
        "Crafted for Those Who Expect the Finest",
    ],
}

BODY_COPY = {
    "Economy": [
        "Discover {destination} with fares designed for the smart traveller. "
        "Every seat includes {amenity}, so you arrive ready to explore.",
        "From {origin} to {destination} — your journey begins the moment you board. "
        "Enjoy {amenity} and seamless connections across our global network.",
        "Life is short. The world is wide. Fly to {destination} and make every "
        "moment count. Book now and enjoy {amenity} on every flight.",
    ],
    "Premium Economy": [
        "Stretch out and settle in on your journey to {destination}. "
        "Premium Economy offers {amenity} — all the space you need to "
        "arrive feeling your best.",
        "The middle ground between comfort and value. Fly to {destination} "
        "with extra legroom, {amenity} and a dining experience designed "
        "to delight.",
        "Because you work hard enough. Treat yourself to Premium Economy "
        "on your next trip to {destination}. Enjoy {amenity} and arrive "
        "refreshed and ready.",
    ],
    "Business": [
        "Your journey to {destination} begins in Business Class. "
        "Lie-flat beds, {amenity} and a dedicated cabin crew ensure "
        "you arrive at your best.",
        "From the moment you board to the moment you land in {destination}, "
        "every detail is crafted for the business traveller. "
        "Experience {amenity} at altitude.",
        "Close the deal before you even land. Business Class to {destination} "
        "gives you the space to work, rest and arrive with your edge intact. "
        "Featuring {amenity}.",
    ],
    "First": [
        "A private suite awaits you on your journey to {destination}. "
        "First Class is more than a seat — it is an experience defined "
        "by {amenity} and flawless personal service.",
        "The world's finest destinations deserve the world's finest journey. "
        "Fly First Class to {destination} and experience {amenity} "
        "at its most exquisite.",
        "For those who accept nothing but the finest. Your journey to "
        "{destination} in First Class features {amenity}, curated dining "
        "and a level of care that redefines luxury travel.",
    ],
}

AMENITIES = {
    "Economy": [
        "complimentary meals",
        "in-flight entertainment",
        "generous baggage allowance",
        "Wi-Fi connectivity",
        "a curated selection of movies and music",
    ],
    "Premium Economy": [
        "extra legroom seating",
        "priority boarding and baggage",
        "enhanced dining and wider seats",
        "dedicated overhead storage",
        "premium amenity kit",
    ],
    "Business": [
        "lie-flat beds and direct aisle access",
        "chef-curated dining and premium wine selection",
        "dedicated lounge access",
        "priority check-in and fast-track security",
        "noise-cancelling headphones and luxury bedding",
    ],
    "First": [
        "a private enclosed suite",
        "on-demand fine dining from our Michelin-starred menu",
        "exclusive first class lounge and spa access",
        "personalised cabin crew service",
        "luxury pyjamas and bespoke amenity collections",
    ],
}

CTAS = {
    "Champions": [
        "Book Your Next Journey",
        "Explore Exclusive Member Rates",
        "Claim Your Priority Access",
    ],
    "Loyal Travellers": [
        "Book Now and Earn Miles",
        "Accelerate Your Status",
        "Reserve Your Seat Today",
    ],
    "At-Risk Frequent": [
        "Return and Save — Limited Time Offer",
        "Your Miles Are Waiting — Book Now",
        "Exclusive Offer: Just for You",
    ],
    "Occasional Flyers": [
        "Discover Where We Fly",
        "Start Your Adventure",
        "Find Your Perfect Destination",
    ],
    "Dormant Members": [
        "We Have Missed You — Special Welcome Back Rate",
        "Reactivate Your Miles Today",
        "Come Back and Save 20%",
    ],
}

SUBJECT_LINES = {
    "Economy": [
        "{destination} is calling - fares from HK${fare:,}",
        "Your next getaway: {destination} at an unbeatable price",
        "Limited seats to {destination} — grab yours now",
        "Fly to {destination} this {month}",
    ],
    "Premium Economy": [
        "Upgrade your {destination} experience — Premium Economy awaits",
        "More comfort, more value: {destination} in Premium Economy",
        "You deserve more room — fly Premium Economy to {destination}",
        "{destination} in style - Premium Economy from HK${fare:,}",
    ],
    "Business": [
        "Arrive ready: Business Class to {destination}",
        "Your {destination} suite awaits — Business Class",
        "Work. Rest. Land in {destination} at your best.",
        "Business Class to {destination} - exclusive member fare",
    ],
    "First": [
        "An invitation to First Class — {destination} awaits",
        "The finest journey to {destination} — First Class",
        "Your private suite to {destination} is ready",
        "First Class. {destination}. An experience unlike any other.",
    ],
}

DESTINATIONS = {
    "LHR": "London",
    "JFK": "New York",
    "SYD": "Sydney",
    "NRT": "Tokyo",
    "SIN": "Singapore",
    "LAX": "Los Angeles",
    "YVR": "Vancouver",
    "FRA": "Frankfurt",
    "CDG": "Paris",
    "DXB": "Dubai",
    "BKK": "Bangkok",
    "TPE": "Taipei",
    "ICN": "Seoul",
    "MEL": "Melbourne",
    "MNL": "Manila",
}

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


# ── Core Generator Functions ──────────────────────────────────────

def generate_headline(cabin: str) -> str:
    """Pick a random headline for the given cabin class."""
    options = HEADLINES.get(cabin, HEADLINES["Economy"])
    return random.choice(options)


def generate_body_copy(cabin: str, route: str) -> str:
    """
    Generate body copy for a given cabin and route.
    Fills in destination and amenity placeholders dynamically.
    """
    dest_code   = route.split("-")[-1] if "-" in route else route
    destination = DESTINATIONS.get(dest_code, dest_code)
    origin      = route.split("-")[0] if "-" in route else "your city"
    amenity     = random.choice(AMENITIES.get(cabin, AMENITIES["Economy"]))

    templates = BODY_COPY.get(cabin, BODY_COPY["Economy"])
    template  = random.choice(templates)

    return template.format(
        destination=destination,
        origin=origin,
        amenity=amenity,
    )


def generate_subject_line(cabin: str, route: str, base_fare: int) -> str:
    """
    Generate an email subject line.
    Fills in destination, fare and month placeholders.
    """
    dest_code   = route.split("-")[-1] if "-" in route else route
    destination = DESTINATIONS.get(dest_code, dest_code)

    import datetime
    current_month = MONTHS[datetime.datetime.now().month - 1]

    templates = SUBJECT_LINES.get(cabin, SUBJECT_LINES["Economy"])
    template  = random.choice(templates)

    return template.format(
        destination=destination,
        fare=base_fare,
        month=current_month,
    )


def generate_cta(segment: str) -> str:
    """Pick a CTA appropriate for the customer segment."""
    options = CTAS.get(segment, CTAS["Occasional Flyers"])
    return random.choice(options)


def generate_full_copy_set(
    cabin:   str,
    route:   str,
    segment: str,
    base_fare: int = 5000,
    n_variations: int = 3,
) -> pd.DataFrame:
    """
    Generate N complete copy variations for a
    cabin × route × segment combination.

    Returns a DataFrame with one row per variation containing:
      headline, body_copy, subject_line, cta, copy_id
    """
    records = []
    for i in range(n_variations):
        records.append({
            "copy_id":      f"COPY-{cabin[:3].upper()}-{i+1:02d}",
            "cabin":        cabin,
            "route":        route,
            "segment":      segment,
            "headline":     generate_headline(cabin),
            "body_copy":    generate_body_copy(cabin, route),
            "subject_line": generate_subject_line(cabin, route, base_fare),
            "cta":          generate_cta(segment),
            "base_fare":    base_fare,
        })

    return pd.DataFrame(records)


def score_subject_lines(subject_lines: list) -> pd.DataFrame:
    """
    Score a list of subject lines on predicted open-rate drivers.
    Uses heuristic rules based on email marketing best practices.

    Scoring criteria:
      - Has emoji              +10 points
      - Has number/fare        +8 points
      - Has destination name   +5 points
      - Length 40-60 chars     +10 points (optimal)
      - Length < 30 chars      -5 points (too short)
      - Length > 70 chars      -8 points (too long)
      - Has urgency word       +7 points
      - Has personalisation    +6 points
    """
    urgency_words = [
        "limited", "exclusive", "now", "today",
        "last chance", "hurry", "only", "special"
    ]
    personalisation_words = [
        "you", "your", "member", "invite",
        "just for you", "waiting"
    ]

    records = []
    for sl in subject_lines:
        score = 50   # Base score
        sl_lower = sl.lower()

        # Emoji check
        has_emoji = any(ord(c) > 127 for c in sl)
        if has_emoji:
            score += 10

        # Number check
        has_number = any(c.isdigit() for c in sl)
        if has_number:
            score += 8

        # Destination check
        has_dest = any(
            dest.lower() in sl_lower
            for dest in DESTINATIONS.values()
        )
        if has_dest:
            score += 5

        # Length score
        length = len(sl)
        if 40 <= length <= 60:
            score += 10
        elif length < 30:
            score -= 5
        elif length > 70:
            score -= 8

        # Urgency
        has_urgency = any(w in sl_lower for w in urgency_words)
        if has_urgency:
            score += 7

        # Personalisation
        has_personal = any(w in sl_lower for w in personalisation_words)
        if has_personal:
            score += 6

        # Add small random variation (simulates A/B test noise)
        score += int(rng.integers(-3, 4))
        score  = max(0, min(100, score))

        records.append({
            "subject_line":        sl,
            "predicted_open_rate": round(score * 0.45 / 100, 4),
            "score":               score,
            "length":              length,
            "has_emoji":           has_emoji,
            "has_number":          has_number,
            "has_urgency":         has_urgency,
        })

    return (
        pd.DataFrame(records)
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )


def generate_multilingual_variants(
    headline: str,
    destination: str,
) -> pd.DataFrame:
    """
    Generate simplified multilingual headline variants.
    Uses pre-built translations for key markets.
    No translation API needed — template based.
    """
    templates = {
        "English":            headline,
        "Traditional Chinese":"探索{dest}的精彩旅程，立即預訂",
        "Simplified Chinese": "探索{dest}的精彩旅程，立即预订",
        "Japanese":           "{dest}への旅を今すぐ予約",
        "Korean":             "{dest}로의 여행을 지금 예약하세요",
        "Thai":               "จองเที่ยวบินไป{dest}วันนี้",
    }

    records = []
    for language, template in templates.items():
        try:
            text = template.format(dest=destination)
        except Exception:
            text = template

        records.append({
            "language": language,
            "headline": text,
            "market":   language,
        })

    return pd.DataFrame(records)
