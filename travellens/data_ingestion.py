"""
data_ingestion.py — Online Travel Data Fetcher for TravelLens

Fetches travel destinations dynamically from public APIs:
  - Wikivoyage API  (primary source — travel-focused content)
  - Wikipedia API   (supplemental descriptions)

No API keys required for these sources.
"""

import logging
import re
import time
from typing import Any

import requests

from embeddings import embed_destinations
from endee_client import get_endee_client

logger = logging.getLogger(__name__)

WIKIVOYAGE_API = "https://en.wikivoyage.org/w/api.php"
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"

HEADERS = {
    "User-Agent": "TravelLens/1.0 (travel-ai-project; contact@travellens.ai)"
}

# Travel destination categories to search per type
DESTINATION_QUERIES: dict[str, list[str]] = {
    "mountain": [
        "mountain destinations India",
        "Himalayan trekking destinations",
        "mountain travel Switzerland",
        "mountain destinations Norway",
        "mountain destinations Nepal",
        "Andes mountain destinations Peru",
        "Rocky Mountains destinations USA",
        "mountain destinations Japan",
        "Alps destinations Austria",
        "mountain destinations New Zealand",
    ],
    "beach": [
        "beach destinations Thailand",
        "tropical beach destinations Maldives",
        "beach destinations Bali Indonesia",
        "beach destinations Greece",
        "beach destinations Caribbean",
        "beach destinations Australia",
        "beach destinations Sri Lanka",
        "beach destinations Philippines",
        "beach destinations Brazil",
        "beach destinations Mexico",
    ],
    "city": [
        "city travel Tokyo Japan",
        "city travel Paris France",
        "city travel New York USA",
        "city travel Dubai UAE",
        "city travel Singapore",
        "city travel Istanbul Turkey",
        "city travel Barcelona Spain",
        "city travel Mumbai India",
        "city travel Amsterdam Netherlands",
        "city travel Prague Czech Republic",
    ],
    "nature": [
        "nature destinations Amazon rainforest",
        "nature destinations Patagonia",
        "national park destinations Kenya",
        "nature destinations Iceland",
        "national park destinations Costa Rica",
        "nature destinations Yellowstone USA",
        "nature destinations Bhutan",
        "nature destinations Madagascar",
        "nature destinations New Zealand nature",
        "nature destinations Canadian Rockies",
    ],
    "cultural": [
        "cultural travel Kyoto Japan",
        "cultural heritage Machu Picchu Peru",
        "cultural travel Rajasthan India",
        "cultural destinations Petra Jordan",
        "cultural travel Rome Italy",
        "cultural destinations Angkor Wat Cambodia",
        "cultural travel Egypt pyramids",
        "cultural travel Morocco",
        "cultural destinations Varanasi India",
        "cultural travel Mexico City",
    ],
}

# Activity mapping per destination type
TYPE_ACTIVITIES: dict[str, list[str]] = {
    "mountain": ["trekking", "hiking", "skiing", "paragliding", "camping", "mountaineering"],
    "beach": ["swimming", "snorkeling", "scuba diving", "surfing", "beach sports", "island hopping"],
    "city": ["sightseeing", "museums", "food tours", "shopping", "nightlife", "architecture tours"],
    "nature": ["wildlife safari", "bird watching", "nature walks", "photography", "eco-tourism", "camping"],
    "cultural": ["temple visits", "heritage tours", "traditional cuisine", "local festivals", "art galleries", "handicraft shopping"],
}

# Tag mapping per destination type
TYPE_TAGS: dict[str, list[str]] = {
    "mountain": ["mountain", "snow", "adventure", "nature", "trekking", "altitude"],
    "beach": ["beach", "ocean", "tropical", "relaxation", "water-sports", "island"],
    "city": ["urban", "culture", "food", "shopping", "modern", "cosmopolitan"],
    "nature": ["wildlife", "forest", "wilderness", "eco-tourism", "scenic", "national-park"],
    "cultural": ["heritage", "history", "temples", "traditions", "art", "architecture"],
}

SEASON_DEFAULTS: dict[str, str] = {
    "mountain": "May to September",
    "beach": "November to April",
    "city": "Year-round",
    "nature": "June to October",
    "cultural": "October to March",
}


def _extract_country_from_text(title: str, summary: str) -> str:
    """Heuristically extract country name from title or first sentence."""
    country_keywords = [
        "India", "Japan", "France", "Italy", "USA", "United States", "Thailand",
        "Indonesia", "Greece", "Switzerland", "Norway", "Nepal", "Peru", "Brazil",
        "Australia", "Sri Lanka", "Philippines", "Mexico", "Canada", "Iceland",
        "New Zealand", "Kenya", "Costa Rica", "Jordan", "Cambodia", "Egypt",
        "Morocco", "Maldives", "Caribbean", "Turkey", "Spain", "Netherlands",
        "Czech Republic", "UAE", "Singapore", "United Kingdom", "Germany",
        "Austria", "Bhutan", "Madagascar", "Chile", "Argentina",
    ]
    combined = title + " " + summary[:200]
    for country in country_keywords:
        if country.lower() in combined.lower():
            return country
    return "International"


def _clean_text(text: str) -> str:
    """Remove wiki markup, references, and extra whitespace."""
    text = re.sub(r"\{\{[^}]+\}\}", "", text)
    text = re.sub(r"\[\[([^\]|]+\|)?([^\]]+)\]\]", r"\2", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"={2,}[^=]+=+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _truncate(text: str, max_chars: int = 400) -> str:
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_period = truncated.rfind(".")
    if last_period > max_chars // 2:
        return truncated[: last_period + 1]
    return truncated.rstrip() + "..."


def fetch_wikivoyage_destinations(query: str, dest_type: str) -> list[dict[str, Any]]:
    """Search Wikivoyage and parse destination articles."""
    destinations: list[dict[str, Any]] = []

    # Step 1: Search Wikivoyage for articles matching the query
    search_params: dict[str, Any] = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": 5,
        "srprop": "snippet",
        "format": "json",
    }
    try:
        resp = requests.get(WIKIVOYAGE_API, params=search_params, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        search_results = resp.json().get("query", {}).get("search", [])
    except Exception as exc:
        logger.warning("Wikivoyage search failed for '%s': %s", query, exc)
        return []

    for item in search_results:
        title = item.get("title", "")
        if not title:
            continue

        # Step 2: Fetch the article extract for description
        extract_params: dict[str, Any] = {
            "action": "query",
            "titles": title,
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
            "exsentences": 6,
            "format": "json",
        }
        try:
            ext_resp = requests.get(WIKIVOYAGE_API, params=extract_params, headers=HEADERS, timeout=15)
            ext_resp.raise_for_status()
            pages = ext_resp.json().get("query", {}).get("pages", {})
            page = next(iter(pages.values()), {})
            extract = page.get("extract", "") or ""
            extract = _clean_text(extract)
        except Exception as exc:
            logger.warning("Wikivoyage extract failed for '%s': %s", title, exc)
            extract = _clean_text(item.get("snippet", ""))

        if len(extract) < 30:
            continue

        description = _truncate(extract, 400)
        country = _extract_country_from_text(title, description)

        dest: dict[str, Any] = {
            "name": title,
            "country": country,
            "region": country,
            "type": dest_type,
            "description": description,
            "activities": TYPE_ACTIVITIES.get(dest_type, ["sightseeing", "photography"]),
            "best_season": SEASON_DEFAULTS.get(dest_type, "Year-round"),
            "tags": TYPE_TAGS.get(dest_type, ["travel", "destination"]),
            "source": "wikivoyage",
        }
        destinations.append(dest)
        time.sleep(0.3)  # polite rate-limiting

    return destinations


def fetch_wikipedia_supplement(name: str) -> str:
    """Fetch a short Wikipedia description to supplement Wikivoyage data."""
    params: dict[str, Any] = {
        "action": "query",
        "titles": name,
        "prop": "extracts",
        "exintro": True,
        "explaintext": True,
        "exsentences": 4,
        "format": "json",
    }
    try:
        resp = requests.get(WIKIPEDIA_API, params=params, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
        page = next(iter(pages.values()), {})
        return _clean_text(page.get("extract", ""))
    except Exception:
        return ""


def deduplicate(destinations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove destinations with the same name (case-insensitive)."""
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for d in destinations:
        key = d["name"].lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(d)
    return unique


def assign_ids(destinations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Assign sequential integer IDs (1-based)."""
    for idx, dest in enumerate(destinations, start=1):
        dest["id"] = idx
    return destinations


def run_ingestion(max_destinations: int = 80) -> int:
    """
    Main ingestion pipeline:
    1. Fetch destinations from Wikivoyage for all types
    2. Deduplicate
    3. Assign IDs
    4. Generate embeddings
    5. Upsert to Endee

    Returns the count of ingested destinations.
    """
    logger.info("=== TravelLens Data Ingestion Started ===")
    all_destinations: list[dict[str, Any]] = []

    for dest_type, queries in DESTINATION_QUERIES.items():
        logger.info("Fetching [%s] destinations...", dest_type)
        type_results: list[dict[str, Any]] = []
        for query in queries:
            if len(all_destinations) + len(type_results) >= max_destinations:
                break
            results = fetch_wikivoyage_destinations(query, dest_type)
            type_results.extend(results)
            logger.info("  '%s' → %d results", query, len(results))

        all_destinations.extend(type_results)
        logger.info("[%s] subtotal: %d", dest_type, len(type_results))

        if len(all_destinations) >= max_destinations:
            break

    logger.info("Total raw fetched: %d", len(all_destinations))

    # Deduplicate
    destinations = deduplicate(all_destinations)
    logger.info("After deduplication: %d unique destinations", len(destinations))

    # Limit
    destinations = destinations[:max_destinations]

    # Assign IDs
    destinations = assign_ids(destinations)

    # Generate embeddings
    logger.info("Generating semantic embeddings...")
    destinations = embed_destinations(destinations)

    # Upsert to Endee
    client = get_endee_client()
    client.get_or_create_index()

    records = [
        {
            "id": str(d["id"]),
            "vector": d["vector"],
            "meta": {
                "id": d["id"],
                "name": d["name"],
                "country": d["country"],
                "region": d.get("region", d["country"]),
                "type": d["type"],
                "description": d["description"],
                "activities": d.get("activities", []),
                "best_season": d.get("best_season", "Year-round"),
                "tags": d.get("tags", []),
                "source": d.get("source", "wikivoyage"),
            },
        }
        for d in destinations
    ]

    client.upsert_destinations(records)
    logger.info("=== Ingestion Complete: %d destinations stored in Endee ===", len(destinations))
    return len(destinations)
