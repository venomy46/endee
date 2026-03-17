"""
seed_data.py — TravelLens Data Seeding Entry Point

Run this script ONCE after starting Endee to populate the vector database
with travel destination data fetched from public APIs.

Usage:
    python seed_data.py
"""

import logging
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger("seed_data")


def wait_for_endee(retries: int = 10, delay: float = 3.0) -> bool:
    """Poll Endee health endpoint until it's ready."""
    import requests
    import os
    from dotenv import load_dotenv

    load_dotenv()
    endee_url = os.getenv("ENDEE_URL", "http://localhost:8080")

    logger.info("Waiting for Endee at %s...", endee_url)
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(f"{endee_url}/api/v1/indices", timeout=5)
            if resp.status_code in (200, 404):  # 404 = no indices yet, but server is up
                logger.info("✅ Endee is ready!")
                return True
        except Exception:
            pass
        logger.info("  Attempt %d/%d — not ready yet, waiting %.0fs...", attempt, retries, delay)
        time.sleep(delay)
    return False


def main() -> None:
    logger.info("=" * 60)
    logger.info("  TravelLens — Data Seeding Pipeline")
    logger.info("=" * 60)

    # 1. Wait for Endee to be healthy
    if not wait_for_endee():
        logger.error("❌ Endee did not become ready. Make sure Docker is running:")
        logger.error("   docker compose up -d")
        sys.exit(1)

    # 2. Run ingestion
    from data_ingestion import run_ingestion

    start = time.time()
    count = run_ingestion(max_destinations=80)
    elapsed = time.time() - start

    logger.info("=" * 60)
    logger.info("  ✅ Seeding Complete!")
    logger.info("  📍 %d travel destinations stored in Endee", count)
    logger.info("  ⏱  Time taken: %.1f seconds", elapsed)
    logger.info("=" * 60)
    logger.info("  Next step: python app.py")
    logger.info("  Then open: http://localhost:8000")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
