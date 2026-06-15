# Semantic search demo

Seed a vecvec collection with embeddings from a **local Ollama model**, then run
natural-language semantic search against it. Two commands, `seed` and `query`,
with the embedder configured in [`config.json`](./config.json).

No npm install needed — pure Node (18+), uses built-in `fetch`.

## Prerequisites

```sh
# 1. a running vecvec server (REST on :6333)
cargo run -p vecvec-server          # from the repo root

# 2. Ollama, with the embedding model pulled
ollama serve                        # if not already running
ollama pull qwen3:4b                # or any model you like
```

> Any Ollama model works — the embedding dimension is detected at runtime and the
> collection is created to match. Swap models via `config.json` or `EMBED_MODEL`.

## Usage

```sh
cd demos/semantic-search

# 1. seed: embed corpus.json and load it into vecvec
node seed.mjs
#   or: npm run seed

# 2. query: one-shot
node query.mjs "how does a database survive a crash?"
#   or: npm run query -- "how does a database survive a crash?"

# 2b. query: interactive REPL (no argument)
node query.mjs
```

Seed your own data by passing a corpus file (a JSON array of
`{ "text": "...", "category": "..." }`):

```sh
node seed.mjs path/to/my-corpus.json
```

## Configuration

[`config.json`](./config.json) — every field can be overridden by an env var:

| Config                  | Env var           | Default                  |
| ----------------------- | ----------------- | ------------------------ |
| `embedder.model`        | `EMBED_MODEL`     | `qwen3:4b`               |
| `embedder.baseUrl`      | `OLLAMA_URL`      | `http://127.0.0.1:11434` |
| `embedder.batchSize`    | `EMBED_BATCH`     | `16`                     |
| `vecvec.url`            | `VECVEC_REST_URL` | `http://127.0.0.1:6333`  |
| `vecvec.collection`     | `COLLECTION`      | `demo-semantic`          |
| `metric`                | `METRIC`          | `cosine`                 |
| `topK`                  | `TOP_K`           | `5`                      |

```sh
EMBED_MODEL=nomic-embed-text TOP_K=8 node query.mjs "ripples in spacetime"
```

## How it works

- **seed.mjs** embeds every document via Ollama's `/api/embed`, detects the vector
  dimension, (re)creates the collection, then upserts each vector (with the original
  text in its `payload`) **followed by a commit per vector** — so the collection
  timeline gets one version per document (`genesis` … `seed`).
- **query.mjs** embeds your question with the same model, runs an HNSW
  nearest-neighbour `query`, then fetches each hit's payload to show the source
  text. Vectors are L2-normalized client-side when the metric is `cosine`.
