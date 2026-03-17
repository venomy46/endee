"""
endee_client.py — Endee Vector Database Client for TravelLens

Wraps the official `endee` Python SDK to manage the travel_destinations index.
"""

import os
import logging
from typing import Any

from dotenv import load_dotenv
from endee import Endee, Precision  # type: ignore[import-untyped]

load_dotenv()

logger = logging.getLogger(__name__)

INDEX_NAME = "travel_destinations"
DIMENSION = 384  # all-MiniLM-L6-v2 output dimension


class EndeeVectorClient:
    """High-level client for Endee vector database operations."""

    def __init__(self) -> None:
        self._url: str = os.getenv("ENDEE_URL", "http://localhost:8080")
        self._auth_token: str = os.getenv("ENDEE_AUTH_TOKEN", "")
        self._client: Endee = self._connect()
        self._index: Any = None

    def _connect(self) -> Endee:
        """Create and configure the Endee SDK client."""
        if self._auth_token:
            client = Endee(self._auth_token)
        else:
            client = Endee()
        client.set_base_url(f"{self._url}/api/v1")
        logger.info("Endee client connected to %s", self._url)
        return client

    def get_or_create_index(self) -> Any:
        """Idempotently create the travel_destinations index."""
        try:
            self._index = self._client.get_index(name=INDEX_NAME)
            logger.info("Loaded existing index: %s", INDEX_NAME)
        except Exception:
            logger.info("Creating new index: %s", INDEX_NAME)
            self._client.create_index(
                name=INDEX_NAME,
                dimension=DIMENSION,
                space_type="cosine",
                precision=Precision.INT8,
            )
            self._index = self._client.get_index(name=INDEX_NAME)
            logger.info("Created index: %s (dim=%d, cosine, INT8)", INDEX_NAME, DIMENSION)
        return self._index

    def upsert_destinations(self, records: list[dict[str, Any]]) -> None:
        """
        Batch upsert destination vectors to Endee.

        Each record must contain:
          - id:     str or int (converted to str)
          - vector: list[float] with 384 dimensions
          - meta:   dict with all destination metadata
        """
        if self._index is None:
            self.get_or_create_index()

        vectors = [
            {
                "id": str(r["id"]),
                "vector": r["vector"],
                "meta": r["meta"],
            }
            for r in records
        ]

        batch_size = 50
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            self._index.upsert(batch)
            logger.info("Upserted batch %d/%d (%d vectors)", i // batch_size + 1, (len(vectors) - 1) // batch_size + 1, len(batch))

    def search(self, query_vector: list[float], top_k: int = 5) -> list[dict[str, Any]]:
        """
        Perform a semantic vector search.

        Returns a list of dicts with:
          - id:    document id
          - score: cosine similarity score
          - meta:  destination metadata payload
        """
        if self._index is None:
            self.get_or_create_index()

        results = self._index.query(vector=query_vector, top_k=top_k)

        destinations = []
        for r in results:
            entry: dict[str, Any] = {"id": r.get("id", ""), "score": r.get("similarity", 0.0)}
            entry.update(r.get("meta", {}))
            destinations.append(entry)

        return destinations

    def get_by_id(self, destination_id: str) -> dict[str, Any] | None:
        """Fetch a single destination by its vector ID."""
        if self._index is None:
            self.get_or_create_index()
        try:
            result = self._index.get(id=destination_id)
            if result:
                meta: dict[str, Any] = result.get("meta", {})
                meta["id"] = destination_id
                return meta
        except Exception as exc:
            logger.warning("Could not fetch id=%s: %s", destination_id, exc)
        return None

    def health_check(self) -> bool:
        """Return True if Endee is reachable and the index exists."""
        try:
            self.get_or_create_index()
            return True
        except Exception as exc:
            logger.error("Endee health check failed: %s", exc)
            return False


# Singleton instance
_client_instance: EndeeVectorClient | None = None


def get_endee_client() -> EndeeVectorClient:
    """Return the shared EndeeVectorClient singleton."""
    global _client_instance
    if _client_instance is None:
        _client_instance = EndeeVectorClient()
    return _client_instance
