# 🌍 TravelLens — AI Travel Discovery Engine

> Semantic search + RAG travel advisor powered by **Endee Vector DB** + **Groq AI** + **SentenceTransformers**

---

## Architecture

```
User Query (natural language)
        │
        ▼
  SentenceTransformers           ← all-MiniLM-L6-v2 (384-dim)
  (generate query embedding)
        │
        ▼
  Endee Vector Database          ← semantic similarity search
  (cosine, INT8, indexed)
        │
        ▼
  Top-K Destinations             ← retrieved from Wikivoyage / Wikipedia
        │
        ▼
  Groq AI — RAG                ← context + query → travel advice
        │
        ▼
  FastAPI Response               ← results + ai_travel_advice
        │
        ▼
  Browser UI                     ← destination cards + AI insight
```

### Data Flow for Ingestion

```
Wikivoyage API + Wikipedia API
        │
        ▼
  data_ingestion.py              ← fetch, clean, normalize
        │
        ▼
  embeddings.py                  ← SentenceTransformer encode
        │
        ▼
  Endee Vector Index             ← upsert vectors + metadata
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Vector Database | **Endee** (Docker, HTTP API, Python SDK) |
| Embeddings | **sentence-transformers** `all-MiniLM-L6-v2` (384-dim) |
| AI Generation | **Groq AI** (OpenAI-compatible, `llama-3.3-70b-versatile`) |
| Backend | **FastAPI** + **Uvicorn** |
| Frontend | HTML + CSS (dark glassmorphism) + Vanilla JS |
| Data Sources | **Wikivoyage API** + **Wikipedia API** (public, free) |

---

## Project Structure

```
travelens/
├── docker-compose.yml      # Endee vector database
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
│
├── endee_client.py         # Endee SDK wrapper (index, upsert, search)
├── embeddings.py           # SentenceTransformer embedding generation
├── data_ingestion.py       # Wikivoyage + Wikipedia data fetcher
├── seed_data.py            # Runs the ingestion pipeline
├── app.py                  # FastAPI backend + Groq AI RAG
│
├── frontend/
│   ├── index.html          # Travel discovery UI
│   ├── style.css           # Dark glassmorphism theme
│   └── app.js              # Search, cards, itinerary modal
│
└── README.md
```

---

## Setup

### 1. Prerequisites
- **Docker Desktop** — [docker.com](https://www.docker.com/products/docker-desktop)
- **Python 3.11+**
- **Groq API key** — [console.groq.com](https://console.groq.com)

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### 3. Start Endee
```bash
docker compose up -d
```
Endee runs at `http://localhost:8080`

### 4. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 5. Seed Travel Data
```bash
python seed_data.py
```
Fetches ~80 destinations from Wikivoyage and Wikipedia, generates embeddings, and stores them in Endee.

> Takes ~3–5 minutes (API fetching + embedding generation)

### 6. Start the API Server
```bash
python app.py
```

### 7. Open in Browser
```
http://localhost:8000
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET`  | `/api/health` | Server + Endee status |
| `POST` | `/api/search` | Semantic search + Groq AI advice |
| `GET`  | `/api/destination/{id}` | Get destination by ID |
| `GET`  | `/api/recommend/{id}` | Similar destinations |
| `POST` | `/api/itinerary` | AI-generated trip itinerary |

### Example Search Request
```bash
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "peaceful mountain places in India", "top_k": 5}'
```

### Example Response
```json
{
  "query": "peaceful mountain places in India",
  "results": [
    {
      "id": "1",
      "name": "Manali",
      "country": "India",
      "type": "mountain",
      "description": "Himalayan destination known for snow and trekking...",
      "activities": ["trekking", "skiing", "paragliding"],
      "best_season": "May to September",
      "score": 0.91
    }
  ],
  "ai_travel_advice": "Manali and Auli are excellent peaceful mountain destinations...",
  "total": 5
}
```

---

## How Endee Vector Database is Used

1. **Index creation**: `travel_destinations` index with `dimension=384`, `space_type="cosine"`, `precision=INT8`
2. **Data storage**: Each destination is stored as a vector + metadata payload (name, country, description, activities, etc.)
3. **Semantic search**: User queries are embedded with the same model and searched using cosine similarity in Endee
4. **Python SDK**: Uses the official `endee` package (`pip install endee`)

## How Groq AI Enables RAG

1. **Retrieve**: Top-K destinations are fetched from Endee based on the query embedding
2. **Augment**: A context block is built from the retrieved destinations
3. **Generate**: Groq AI receives the context + user query and generates expert travel advice

Groq API is accessed via its OpenAI-compatible endpoint (`https://api.groq.com/openai/v1`).

---

## Example Queries to Try

- `"peaceful mountain places in India"`
- `"romantic beach destinations in Asia"`
- `"budget backpacking places in Europe"`
- `"adventure nature destinations"`
- `"cultural heritage sites in Asia"`
- `"snowy ski destinations Europe"`

---

## License

MIT
