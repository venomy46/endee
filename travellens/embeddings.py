"""
embeddings.py — SentenceTransformer Embedding Generator for TravelLens

Uses all-MiniLM-L6-v2 (384-dim) to embed travel destination text.
"""

import logging
from functools import lru_cache
from typing import Any

from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    """Load and cache the SentenceTransformer model (downloaded once)."""
    logger.info("Loading SentenceTransformer model: %s", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)
    logger.info("Model loaded successfully (dim=384)")
    return model


def generate_embedding(text: str) -> list[float]:
    """Generate a 384-dimensional embedding for a given text string."""
    model = _load_model()
    vector: list[float] = model.encode(text, normalize_embeddings=True).tolist()
    return vector


def build_embedding_text(destination: dict[str, Any]) -> str:
    """
    Build a rich text representation from a destination dict for embedding.

    Combines: name, country, region, type, description, activities, tags, best_season.
    Richer text = better semantic representations.
    """
    parts: list[str] = []

    if name := destination.get("name"):
        parts.append(name)

    if country := destination.get("country"):
        parts.append(f"in {country}")

    if region := destination.get("region"):
        parts.append(f"region {region}")

    if dest_type := destination.get("type"):
        parts.append(f"{dest_type} destination")

    if description := destination.get("description"):
        parts.append(description)

    if activities := destination.get("activities"):
        if isinstance(activities, list):
            parts.append("Activities: " + ", ".join(activities))
        else:
            parts.append(f"Activities: {activities}")

    if tags := destination.get("tags"):
        if isinstance(tags, list):
            parts.append("Tags: " + ", ".join(tags))
        else:
            parts.append(f"Tags: {tags}")

    if season := destination.get("best_season"):
        parts.append(f"Best visited: {season}")

    return ". ".join(parts)


def embed_destinations(destinations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Add 'vector' and 'embedding_text' fields to each destination dict.

    Returns the updated list (in-place modification + return).
    """
    model = _load_model()
    texts = [build_embedding_text(d) for d in destinations]
    logger.info("Generating embeddings for %d destinations...", len(texts))
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    for dest, vec, text in zip(destinations, vectors, texts):
        dest["vector"] = vec.tolist()
        dest["embedding_text"] = text
    logger.info("Embeddings generated.")
    return destinations
