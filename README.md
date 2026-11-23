# AI Agent — FastAPI + Vercel

An open-source AI agent you can deploy to a live URL in one command.

It is one small Python service that does four things:

| Capability | What it means |
| --- | --- |
| 💬 Chat | Streaming conversations with an OpenAI-compatible model |
| 🛠️ Tools | The model chooses: calculator, clock, web search, URL reader, document search |
| 📚 RAG | Upload PDF / Markdown / text → answers with citations |
| 🖥️ UI | Single-file chat interface, no build step, served from the CDN |

**Free to run:** defaults to Groq's free tier (no credit card). Works with
OpenRouter, Google Gemini, OpenAI, or a local Ollama server — one env var each.
**Works with zero keys too:** demo mode still answers from your documents.

```
┌──────────────┐      ┌───────────────────────────────┐      ┌──────────────┐
│ public/      │      │ main.py → agentapp.main:app   │      │ Groq /       │
│ index.html   │ ───► │  /api/chat  /api/chat/stream  │ ───► │ OpenRouter / │
│ (CDN static) │      │  /api/documents  /api/health  │      │ Gemini /     │
└──────────────┘      │  agent loop + tools + RAG     │      │ Ollama       │
                      └───────────────────────────────┘      └──────────────┘
```

---

## Quickstart (local)

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux

pip install -r requirements-dev.txt
copy .env.example .env          # Windows (cp on macOS/Linux)
# paste your key into .env, e.g. GROQ_API_KEY=gsk_...

uvicorn agentapp.main:app --reload --port 8000
```

* App → http://localhost:8000
* API docs → http://localhost:8000/api/docs

Get a free key at <https://console.groq.com/keys>. Skip this step if you just
want to see document search working — the app runs in demo mode without keys.

## Deploy to Vercel

### Option A — dashboard (recommended)

1. Push this folder to GitHub.
2. Vercel → **Add New → Project** → import the repo.
3. Framework preset: **auto-detected (FastAPI)**. No build command, no output dir.
4. **Settings → Environment Variables** → add `GROQ_API_KEY` (and optionally
   `GEMINI_API_KEY` for real embeddings).
5. **Deploy.** You get `https://your-project.vercel.app` with UI + API live.

### Option B — CLI

```bash
npm i -g vercel
vercel link
vercel env add GROQ_API_KEY production
vercel --prod
```

### Why it deploys cleanly

Vercel's Python runtime looks for a FastAPI instance named `app` in `main.py`
at the project root — that is exactly what this repo exports. `vercel.json`
raises the function timeout to 60s and bundles `data/docs/`. `public/` is served
as static files, so the UI never costs function invocations.

---

## Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `LLM_PROVIDER` | auto-detect | `groq` \| `openrouter` \| `gemini` \| `openai` \| `ollama` |
| `GROQ_API_KEY` | – | Groq key (free tier) |
| `OPENROUTER_API_KEY` | – | OpenRouter key (has `:free` models) |
| `GEMINI_API_KEY` | – | Gemini chat **and** embeddings |
| `OPENAI_API_KEY` | – | OpenAI key |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Local models |
| `LLM_MODEL` | provider default | Override the model id |
| `LLM_TEMPERATURE` | provider default | Set to enable a fixed temperature |
| `EMBEDDING_BACKEND` | `gemini` if key else `hashing` | `gemini` \| `hashing` |
| `EMBEDDING_MODEL` | `gemini-embedding-001` | Embedding model |
| `TAVILY_API_KEY` | – | Better web search (1000 free/month) |
| `AGENT_API_KEY` | – | Require `X-API-Key` on `/api/*` |
| `ALLOWED_ORIGINS` | `*` | CORS allow-list, comma separated |
| `RATE_LIMIT_PER_MIN` | `30` | Best-effort per-IP limit (`0` = off) |
| `AGENT_MAX_STEPS` | `5` | Max tool round-trips per question |
| `MAX_UPLOAD_MB` | `4` | Upload cap (Vercel body limit ≈ 4.5 MB) |

Provider defaults: `groq → openai/gpt-oss-120b`,
`openrouter → meta-llama/llama-3.3-70b-instruct:free`,
`gemini → gemini-2.5-flash`, `ollama → qwen2.5:7b`.

---

## API

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/api/health` | Liveness + provider + indexed chunk count |
| `GET` | `/api/config` | Public config (provider, model, tools) |
| `POST` | `/api/chat` | `{messages:[{role,content}]}` → `{answer, steps, usage}` |
| `POST` | `/api/chat/stream` | Same body → SSE: `step`, `token`, `reset`, `done`, `error` |
| `GET` | `/api/documents` | List indexed documents |
| `POST` | `/api/documents` | Multipart upload (`file`) |
| `POST` | `/api/documents/text` | `{text, source}` — index raw text |
| `POST` | `/api/documents/search` | `{query, k}` — raw retrieval, no LLM |
| `DELETE` | `/api/documents?source=…` | Remove a document |

```bash
curl -s https://your-app.vercel.app/api/chat \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is 12 times 12?"}]}'
```

```python
import httpx

with httpx.stream("POST", "https://your-app.vercel.app/api/chat/stream",
                  json={"messages": [{"role": "user", "content": "Summarise my docs"}]}) as r:
    for line in r.iter_lines():
        if line.startswith("data:"):
            print(line[5:].strip())
```

---

## Project layout

```
main.py                    Vercel entrypoint (exports app)
vercel.json                maxDuration + bundled data/
requirements.txt           runtime deps (installed by Vercel)
public/index.html          chat UI — no build step
agentapp/
  main.py                  FastAPI app, CORS, static mount
  config.py                all settings from env
  security.py              optional X-API-Key + per-IP rate limit
  schemas.py               pydantic models
  providers/llm.py         one OpenAI-compatible client (all providers)
  providers/embeddings.py  Gemini embeddings + local hashing fallback
  agent/prompts.py         system prompt
  agent/tools.py           5 tools + JSON schemas + SSRF guard
  agent/loop.py            plan → act → answer loop, JSON + SSE
  rag/chunker.py           paragraph-aware chunking
  rag/store.py             in-memory cosine vector store
  rag/service.py           seeding, ingest, search, delete
  routers/                 chat, documents, system
data/docs/                 seed knowledge base (bundled into the deployment)
scripts/fake_provider_check.py   end-to-end check, no API key needed
tests/test_smoke.py        pytest suite (no keys needed)
```

## Tests

```bash
python -m pytest -q                     # 7 tests, demo mode, no network
python scripts/fake_provider_check.py   # full agent loop against a fake provider
```

The second one is the interesting one: it boots a fake OpenAI-compatible server
and walks the real tool-calling and streaming code paths, so you can verify a
change without burning free-tier quota.

## Serverless reality check

Vercel functions are stateless and capped at 60s / ~4.5MB body. This project is
built around that:

* conversation history is sent by the browser on every request;
* `data/docs/` is bundled, so cold starts still have a knowledge base;
* uploads are cached in `/tmp` (survives warm reuse, not cold starts);
* the agent loop is capped at 5 tool steps to stay inside the timeout;
* no numpy / ML libraries, so cold starts stay fast.

For persistent uploads or a large corpus, swap `agentapp/rag/store.py` for
pgvector / Upstash Vector / Qdrant. Only `add`, `search`, `delete_source` are used.

## Extending

Add a tool in three steps, in `agentapp/agent/tools.py`:

```python
async def _weather(args: dict) -> str:
    city = args.get("city", "Chennai")
    return f"Sunny, 31C in {city}"   # call a real API here

TOOLS["get_weather"] = Tool(
    name="get_weather",
    description="Current weather for a city.",
    parameters={"type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]},
    handler=_weather,
)
```

The schema is advertised to the model automatically on the next request.

## Security notes

* Set `AGENT_API_KEY` before sharing a public URL — otherwise anyone can spend
  your tokens.
* `RATE_LIMIT_PER_MIN` is best-effort and per warm instance; use Vercel Firewall
  or a gateway for real limits.
* `fetch_url` refuses private/loopback addresses, but a full SSRF defence should
  also resolve DNS and pin the IP — put it behind an egress proxy if that matters.
* There is no code execution tool by default, on purpose.

## License

MIT — see [LICENSE](LICENSE). Use it, fork it, ship it.
