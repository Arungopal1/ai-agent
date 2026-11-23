# Deployment guide

## Local development

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux
pip install -r requirements-dev.txt
copy .env.example .env        # then paste your API key
uvicorn agentapp.main:app --reload --port 8000
```

Open http://localhost:8000 and the chat UI loads. The API docs are at
http://localhost:8000/api/docs.

## Deploying to Vercel

1. Push the project to a GitHub, GitLab or Bitbucket repository.
2. In Vercel, choose **Add New -> Project** and import the repository.
3. Vercel detects FastAPI automatically because `requirements.txt` lists it and
   `main.py` exports a variable named `app`. No build command, no output
   directory, no framework preset is needed.
4. Add environment variables under **Settings -> Environment Variables**:
   `GROQ_API_KEY` (or `OPENROUTER_API_KEY` / `GEMINI_API_KEY`) and, if you want
   document search with real embeddings, `GEMINI_API_KEY` too.
5. Deploy. The first build takes a couple of minutes; later builds are cached.

Or from the terminal:

```bash
npm i -g vercel
vercel link
vercel env add GROQ_API_KEY
vercel --prod
```

## Serverless limits that shape the design

| Limit | Value on Vercel | What we do about it |
| --- | --- | --- |
| Function duration | 60 s (configured in `vercel.json`) | Agent loop capped at 5 tool steps |
| Request body | ~4.5 MB | Uploads capped at 4 MB |
| Filesystem | read-only except `/tmp` | Seed docs are bundled; uploads are cached in `/tmp` |
| Cold starts | ~1-2 s | No numpy, no heavy ML libraries in the bundle |
| Stateless instances | memory resets | The browser sends the full conversation history |

Everything still works after a cold start because `data/docs/` ships with the
deployment. Uploaded files survive only while an instance stays warm, which is
why the UI warns about it.

## Moving to persistent storage

When you outgrow the in-memory store, replace `agentapp/rag/store.py` with a
pgvector, Upstash Vector or Qdrant adapter. The rest of the code only depends on
three methods: `add`, `search` and `delete_source`.

## Adding tools

1. Write an `async def handler(args: dict) -> str` in `agentapp/agent/tools.py`.
2. Register it in the `TOOLS` dict with a description and a JSON schema.
3. Nothing else changes - the schema is advertised to the model automatically.
