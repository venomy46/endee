"""
app.py — TravelLens FastAPI Backend

Endpoints:
  GET  /api/health
  POST /api/search          — semantic search + Groq RAG travel advice
  GET  /api/destination/{id}
  GET  /api/recommend/{destination_id}
  POST /api/itinerary       — Groq AI multi-day itinerary
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel

from embeddings import generate_embedding
from endee_client import get_endee_client

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("app")

# ─── Groq AI Client ──────────────────────────────────────────────────────────

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"  # Groq model

_groq_client: OpenAI | None = None


def get_groq_client() -> OpenAI:
    global _groq_client
    if _groq_client is None:
        _groq_client = OpenAI(
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
        )
    return _groq_client


# ─── App Lifecycle ────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Starting TravelLens API...")
    client = get_endee_client()
    client.get_or_create_index()
    logger.info("Endee index ready.")
    # Pre-warm the embedding model so first search is instant
    logger.info("Pre-loading embedding model (this may take a moment on first run)...")
    generate_embedding("warm up")
    logger.info("Embedding model ready. TravelLens is fully started!")
    yield
    logger.info("Shutting down TravelLens API.")


# ─── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="TravelLens API",
    description="AI-powered travel discovery using Endee vector DB + Groq AI",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Pydantic Models ──────────────────────────────────────────────────────────


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchResponse(BaseModel):
    query: str
    results: list[dict[str, Any]]
    ai_travel_advice: str
    total: int


class ItineraryRequest(BaseModel):
    destination: str
    days: int = 5
    travel_style: str = "balanced"  # adventure / relaxation / cultural / balanced


# ─── Groq AI Helpers ────────────────────────────────────────────────────────


def build_destination_context(destinations: list[dict[str, Any]]) -> str:
    """Build a structured context block for Groq AI from search results."""
    lines: list[str] = []
    for d in destinations:
        name = d.get("name", "Unknown")
        country = d.get("country", "")
        dest_type = d.get("type", "")
        desc = d.get("description", "")[:200]
        activities = d.get("activities", [])
        act_str = ", ".join(activities[:3]) if activities else "sightseeing"
        lines.append(f"• {name} ({country}) — {dest_type} destination. {desc} Popular activities: {act_str}.")
    return "\n".join(lines)


def get_travel_advice(query: str, destinations: list[dict[str, Any]]) -> str:
    """
    Call Groq AI with retrieved destinations as context (RAG).
    Returns 2–3 sentences of expert travel advice.
    """
    if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
        # Graceful degradation when no API key is configured
        if destinations:
            top = destinations[0]
            return (
                f"{top.get('name', 'This destination')} ({top.get('country', '')}) is an excellent choice for your travel goal. "
                f"It offers {', '.join(top.get('activities', ['great experiences'])[:2])} and is best visited during {top.get('best_season', 'season')}. "
                f"Configure your GROQ_API_KEY in .env for personalized AI travel advice powered by Groq."
            )
        return "No destinations found for your query. Try a broader search like 'beach destinations' or 'mountain travel'."

    context = build_destination_context(destinations)
    prompt = (
        f"You are a world-class travel expert with deep knowledge of global destinations.\n\n"
        f"A traveler is looking for: \"{query}\"\n\n"
        f"Based on these retrieved destinations:\n{context}\n\n"
        f"Provide concise, expert travel advice in 2–3 sentences. "
        f"Suggest the best destination for the traveler's goal and explain why. "
        f"Be specific, inspiring, and practical."
    )

    try:
        groq = get_groq_client()
        response = groq.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a world-class travel advisor. Be concise, specific, and inspiring."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=200,
            temperature=0.7,
        )
        return response.choices[0].message.content or "No advice generated."
    except Exception as exc:
        logger.warning("Groq AI call failed: %s", exc)
        return f"AI travel advice temporarily unavailable. Top pick: {destinations[0].get('name', 'Unknown')} — highly recommended for your trip!"


def generate_itinerary(destination: str, days: int, style: str) -> str:
    """Generate a multi-day travel itinerary using Groq AI."""
    if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
        return f"Sample {days}-day itinerary for {destination}: Day 1: Arrival & local exploration. Day 2-{days-1}: Key attractions. Day {days}: Departure. Configure GROQ_API_KEY for AI-generated itineraries."

    prompt = (
        f"Create a detailed {days}-day travel itinerary for {destination} "
        f"in a {style} travel style. Include morning, afternoon, and evening activities. "
        f"Keep it practical with specific places, food recommendations, and tips. "
        f"Format as Day 1, Day 2, etc."
    )

    try:
        groq = get_groq_client()
        response = groq.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert travel planner. Create detailed, practical itineraries."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1000,
            temperature=0.8,
        )
        return response.choices[0].message.content or "Itinerary generation failed."
    except Exception as exc:
        logger.warning("Groq itinerary generation failed: %s", exc)
        return f"Could not generate itinerary for {destination}. Please try again."


# ─── API Endpoints ────────────────────────────────────────────────────────────


@app.get("/api/health")
async def health_check() -> JSONResponse:
    """Health check for API and Endee connectivity."""
    endee_ok = get_endee_client().health_check()
    groq_configured = bool(GROQ_API_KEY and GROQ_API_KEY != "your_groq_api_key_here")

    return JSONResponse({
        "status": "healthy" if endee_ok else "degraded",
        "endee": "connected" if endee_ok else "disconnected",
        "groq_ai": "configured" if groq_configured else "not_configured (set GROQ_API_KEY in .env)",
        "model": GROQ_MODEL,
        "embedding_model": "all-MiniLM-L6-v2",
        "version": "1.0.0",
    })


@app.post("/api/search", response_model=SearchResponse)
async def search_destinations(request: SearchRequest) -> dict[str, Any]:
    """
    Semantic search over Endee + Groq AI travel advice (RAG).

    Flow:
      1. Embed the user query with SentenceTransformers
      2. Search Endee vector index
      3. Build context from top results
      4. Send context + query to Groq AI
      5. Return results + AI advice
    """
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    logger.info("Search: '%s' (top_k=%d)", query, request.top_k)

    # Step 1: Embed query
    query_vector = generate_embedding(query)

    # Step 2: Vector search in Endee
    client = get_endee_client()
    destinations = client.search(query_vector=query_vector, top_k=request.top_k)

    if not destinations:
        return {
            "query": query,
            "results": [],
            "ai_travel_advice": "No destinations found yet. Run `python seed_data.py` to populate the database.",
            "total": 0,
        }

    # Step 3 + 4: RAG — Groq AI advice
    ai_advice = get_travel_advice(query, destinations)

    return {
        "query": query,
        "results": destinations,
        "ai_travel_advice": ai_advice,
        "total": len(destinations),
    }


@app.get("/api/destination/{destination_id}")
async def get_destination(destination_id: str) -> dict[str, Any]:
    """Retrieve a single destination by its vector ID."""
    client = get_endee_client()
    dest = client.get_by_id(destination_id)
    if not dest:
        raise HTTPException(status_code=404, detail=f"Destination '{destination_id}' not found.")
    return dest


@app.get("/api/recommend/{destination_id}")
async def recommend_similar(destination_id: str, top_k: int = 4) -> dict[str, Any]:
    """
    Find destinations similar to a given one using vector search.
    Fetches the destination's embedding text, re-embeds it, and searches.
    """
    client = get_endee_client()
    dest = client.get_by_id(destination_id)
    if not dest:
        raise HTTPException(status_code=404, detail=f"Destination '{destination_id}' not found.")

    # Build query text from destination metadata for similarity search
    query_text = f"{dest.get('name', '')} {dest.get('type', '')} {dest.get('country', '')} {dest.get('description', '')[:150]}"
    query_vector = generate_embedding(query_text)

    raw: list[dict[str, Any]] = client.search(query_vector=query_vector, top_k=top_k + 1)
    # Exclude the destination itself from results
    similar: list[dict[str, Any]] = [s for s in raw if str(s.get("id", "")) != str(destination_id)][:top_k]  # type: ignore[index]

    return {
        "destination": dest,
        "similar_destinations": similar,
        "total": len(similar),
    }


@app.post("/api/itinerary")
async def create_itinerary(request: ItineraryRequest) -> dict[str, Any]:
    """Generate a Groq AI-powered multi-day travel itinerary."""
    if not request.destination.strip():
        raise HTTPException(status_code=400, detail="Destination cannot be empty.")

    itinerary = generate_itinerary(
        destination=request.destination,
        days=request.days,
        style=request.travel_style,
    )

    return {
        "destination": request.destination,
        "days": request.days,
        "travel_style": request.travel_style,
        "itinerary": itinerary,
    }


# ─── Static Files (Frontend) ──────────────────────────────────────────────────

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    logger.info("Starting TravelLens server at http://localhost:8000")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
