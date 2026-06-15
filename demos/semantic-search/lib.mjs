// Shared helpers for the semantic-search demo: config loading, an Ollama embedder,
// and a thin vecvec REST client. No external dependencies — Node 18+ ships global
// `fetch`, which is all we need here.

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));

/**
 * Load config.json next to this file, then apply environment-variable overrides so
 * you can point the demo at a different model/server without editing the file:
 *
 *   EMBED_MODEL=qwen3:4b OLLAMA_URL=http://localhost:11434 \
 *   VECVEC_REST_URL=http://localhost:6333 COLLECTION=demo node seed.mjs
 */
export function loadConfig() {
  const cfg = JSON.parse(readFileSync(join(HERE, "config.json"), "utf8"));
  const env = process.env;

  cfg.embedder.baseUrl =
    env.OLLAMA_URL ?? env.OLLAMA_BASE_URL ?? cfg.embedder.baseUrl;
  cfg.embedder.model = env.EMBED_MODEL ?? cfg.embedder.model;
  if (env.EMBED_BATCH) cfg.embedder.batchSize = Number(env.EMBED_BATCH);

  cfg.vecvec.url = env.VECVEC_REST_URL ?? cfg.vecvec.url;
  cfg.vecvec.collection = env.COLLECTION ?? cfg.vecvec.collection;

  cfg.metric = env.METRIC ?? cfg.metric;
  if (env.TOP_K) cfg.topK = Number(env.TOP_K);

  return cfg;
}

// ── Embedder (Ollama) ──────────────────────────────────────────────────────

/**
 * Embed an array of strings via Ollama's /api/embed endpoint, in batches.
 * Returns a parallel array of Float32-friendly number[] vectors.
 */
export async function embed(cfg, texts, { onProgress } = {}) {
  const { baseUrl, model, batchSize = 16 } = cfg.embedder;
  const out = [];

  for (let i = 0; i < texts.length; i += batchSize) {
    const batch = texts.slice(i, i + batchSize);
    let res;
    try {
      res = await fetch(`${baseUrl}/api/embed`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model, input: batch }),
      });
    } catch (e) {
      throw new Error(
        `cannot reach Ollama at ${baseUrl} (${e.message}). Is it running?  ollama serve`,
      );
    }

    const text = await res.text();
    if (!res.ok) {
      let msg = text;
      try {
        msg = JSON.parse(text).error ?? text;
      } catch {
        /* keep raw text */
      }
      if (/not found|no such model|pull/i.test(msg)) {
        throw new Error(
          `Ollama model "${model}" is not available (${msg}). Pull it first:  ollama pull ${model}`,
        );
      }
      throw new Error(`Ollama /api/embed → ${res.status}: ${msg}`);
    }

    const json = JSON.parse(text);
    const vecs = json.embeddings;
    if (!Array.isArray(vecs) || vecs.length !== batch.length) {
      throw new Error(
        `unexpected Ollama response: expected ${batch.length} embeddings, got ${vecs?.length}`,
      );
    }
    out.push(...vecs);
    onProgress?.(Math.min(i + batch.length, texts.length), texts.length);
  }
  return out;
}

/** Embed a single string and return one vector. */
export async function embedOne(cfg, text) {
  const [v] = await embed(cfg, [text]);
  return v;
}

/** L2-normalize a vector in place-free fashion (returns a new array). */
export function l2normalize(v) {
  let norm = 0;
  for (const x of v) norm += x * x;
  norm = Math.sqrt(norm);
  if (norm === 0) return v.slice();
  return v.map((x) => x / norm);
}

/** Normalize only when the metric needs it (cosine); otherwise pass through. */
export function prepVector(cfg, v) {
  return cfg.metric === "cosine" ? l2normalize(v) : v;
}

// ── vecvec REST client ─────────────────────────────────────────────────────

export async function vv(cfg, method, path, body) {
  const base = cfg.vecvec.url;
  let res;
  try {
    res = await fetch(base + path, {
      method,
      headers: { "Content-Type": "application/json" },
      body: body ? JSON.stringify(body) : undefined,
    });
  } catch (e) {
    throw new Error(
      `cannot reach vecvec at ${base} (${e.message}). Start it with:  cargo run -p vecvec-server`,
    );
  }
  const text = await res.text();
  let json = null;
  try {
    json = text ? JSON.parse(text) : null;
  } catch {
    json = text; // /healthz returns plain "ok"
  }
  if (!res.ok) {
    throw new Error(`${method} ${path} → ${res.status}: ${json?.error ?? text}`);
  }
  return json;
}

export const vvHealth = (cfg) => vv(cfg, "GET", "/healthz");

/** Drop the collection if it exists, then recreate it fresh with the given dim. */
export async function recreateCollection(cfg, dim) {
  const name = cfg.vecvec.collection;
  await vv(cfg, "DELETE", `/collections/${name}`).catch(() => {});
  await vv(cfg, "POST", `/collections/${name}`, { dim, metric: cfg.metric });
}

/**
 * Upsert points, chunked so each request stays under the server's body-size limit
 * (~2MB). High-dimensional models (e.g. 3072-dim) blow past that in a single POST,
 * so we size each chunk from the vector dimension. Returns aggregated counts/ids.
 */
export async function upsertPoints(cfg, points, { onProgress } = {}) {
  const name = cfg.vecvec.collection;
  const dim = points[0]?.vector.length ?? 1;
  // ~40k floats/request keeps the JSON body well under 2MB for any dimension.
  const chunkSize = Math.max(1, Math.min(512, Math.floor(40000 / dim)));

  let inserted = 0;
  const ids = [];
  for (let i = 0; i < points.length; i += chunkSize) {
    const chunk = points.slice(i, i + chunkSize);
    const r = await vv(cfg, "POST", `/collections/${name}/points`, { points: chunk });
    inserted += r.inserted;
    if (Array.isArray(r.ids)) ids.push(...r.ids);
    onProgress?.(Math.min(i + chunk.length, points.length), points.length);
  }
  return { inserted, ids };
}

export function commit(cfg, message, tag) {
  const name = cfg.vecvec.collection;
  return vv(cfg, "POST", `/collections/${name}/commit`, { message, tag });
}

export function queryVector(cfg, vector, k) {
  const name = cfg.vecvec.collection;
  return vv(cfg, "POST", `/collections/${name}/query`, { vector, k });
}

export function getPoint(cfg, id) {
  const name = cfg.vecvec.collection;
  return vv(cfg, "GET", `/collections/${name}/points/${id}`);
}

export function collectionStats(cfg) {
  const name = cfg.vecvec.collection;
  return vv(cfg, "GET", `/collections/${name}`);
}
