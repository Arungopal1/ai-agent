# Frequently asked questions

**Which LLM providers are supported?**
Groq, OpenRouter, Gemini, OpenAI and any OpenAI-compatible gateway, including a
local Ollama server. Set `LLM_PROVIDER` and the matching API key.

**Which models are free right now?**
On Groq's free plan: `openai/gpt-oss-120b`, `openai/gpt-oss-20b` and
`qwen/qwen3.8-27b`. OpenRouter also has `:free` variants of several open
models. Gemini has a free tier for `gemini-2.5-flash` and
`gemini-embedding-001`.

**Do I need an embedding model?**
Only if you want semantic document search. Without an embedding key the service
falls back to a local hashing vectoriser, which matches keywords but not
meaning. It is fine for a demo, not for production retrieval.

**How many documents can it hold?**
Comfortably a few thousand chunks in memory. Each chunk is roughly 1,200
characters. Beyond that, use a real vector database.

**Does it remember conversations?**
The API is stateless: the browser sends the last 20 messages with every
request. That keeps serverless instances interchangeable and makes the whole
thing horizontally scalable.

**How is it secured?**
Optional `AGENT_API_KEY` requires an `X-API-Key` header, and a best-effort
per-IP rate limit applies to `/api/*`. The `fetch_url` tool refuses private and
loopback addresses to reduce server-side request forgery risk. For a public
deployment, put a real gateway or Vercel Firewall in front.

**Why does the answer cite a file name?**
The agent is instructed to ground RAG answers in retrieved excerpts and cite
the source document in square brackets, so you can verify every claim.

**Can it run a browser or execute code?**
Not by default. Code execution is deliberately omitted because it is a remote
code execution risk on a public endpoint. Add a sandboxed runner if you need it.
