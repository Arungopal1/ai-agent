# About this agent

This project is an open-source AI agent you can deploy in a few minutes.

It is a single Python service that combines four things people usually build
separately:

1. **Chat** - a conversational assistant backed by an OpenAI-compatible model
   (Groq, OpenRouter, Gemini, OpenAI or a local Ollama instance).
2. **Tool calling** - the model can decide to run a calculator, read the
   current time, search the web, fetch a URL, or look inside your documents.
3. **Retrieval augmented generation (RAG)** - upload PDFs, Markdown or text
   files and the agent answers questions with citations to the exact source.
4. **A streaming web UI** - a single HTML file with no build step, served from
   the same deployment.

## Architecture

```
public/index.html      browser UI (static, served by the CDN)
main.py                Vercel entrypoint -> agentapp.main:app
agentapp/
  main.py              FastAPI app, CORS, static mount
  config.py            all settings, read from environment variables
  security.py          optional API key + per-IP rate limit
  providers/llm.py     one OpenAI-compatible client for every provider
  providers/embeddings.py  Gemini embeddings, local hashing fallback
  agent/tools.py       tool registry (5 tools)
  agent/loop.py        the plan -> act -> answer loop, with streaming
  rag/                 chunking, in-memory vector store, ingest service
  routers/             /api/chat, /api/documents, /api/health, /api/config
data/docs/             seed knowledge base, bundled with the deployment
```

## Design decisions

* **One provider interface.** Groq, OpenRouter, Gemini, OpenAI and Ollama all
  speak the OpenAI chat-completions format, so switching providers is one
  environment variable, not a rewrite.
* **Retrieval is a tool.** The model calls `search_documents` only when a
  question looks like it needs the knowledge base. This keeps ordinary chat
  fast and makes citation behaviour explicit.
* **No vector database.** Vectors live in memory and are rebuilt from
  `data/docs/` plus `/tmp` uploads on each cold start. At a few thousand
  chunks this is faster and cheaper than running pgvector, and it means a
  fresh deployment works with zero extra services.
* **Zero-key demo mode.** Without an API key the service still answers using
  document retrieval, so a deployed URL is never a blank error page.
* **Fail soft.** Tool errors, provider errors and bad uploads all return
  readable messages to the model or the user instead of raising 500s.

## Rate limits worth knowing

Free tiers are small. Groq's free plan allows about 30 requests per minute and
1,000 requests per day per model. The service retries once and returns a clear
message when the provider says 429, but for real traffic you should move to a
paid tier or add a queue.
