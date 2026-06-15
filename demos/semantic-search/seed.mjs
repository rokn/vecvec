#!/usr/bin/env node
// Seed a vecvec collection with embeddings produced by a local Ollama model.
//
//   node seed.mjs                 # uses config.json
//   EMBED_MODEL=qwen3:4b node seed.mjs
//   node seed.mjs path/to/corpus.json
//
// The embedding dimension is detected from the model at runtime, so you can swap
// models freely — the collection is (re)created to match.

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import {
  loadConfig,
  embed,
  prepVector,
  vvHealth,
  recreateCollection,
  upsertPoints,
  commit,
  collectionStats,
} from "./lib.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));

// How many documents to seed. The base corpus is small, so we expand it up to
// this count by deriving paraphrase-style variants of each document. Override
// with TARGET_DOCS=… node seed.mjs
const TARGET_DOCS = Number(process.env.TARGET_DOCS ?? 1000);

// Framings used to derive distinct-but-related variants of a base document, so
// every entry embeds to a slightly different point instead of a duplicate.
const FRAMINGS = [
  (t) => t,
  (t) => `In short: ${t}`,
  (t) => `Did you know? ${t}`,
  (t) => `A key takeaway: ${t}`,
  (t) => `Worth remembering — ${t}`,
  (t) => `To put it plainly, ${t.charAt(0).toLowerCase()}${t.slice(1)}`,
  (t) => `Here is the idea: ${t}`,
  (t) => `${t} That detail matters in practice.`,
  (t) => `${t} It comes up often.`,
  (t) => `Consider this: ${t}`,
];

// Expand the base corpus to `count` documents by cycling through framings.
function expandCorpus(base, count) {
  if (base.length >= count) return base.slice(0, count);
  const out = [];
  for (let i = 0; i < count; i++) {
    const doc = base[i % base.length];
    const frame = FRAMINGS[Math.floor(i / base.length) % FRAMINGS.length];
    out.push({ text: frame(doc.text), category: doc.category ?? null });
  }
  return out;
}

async function main() {
  const cfg = loadConfig();
  const corpusPath = process.argv[2] ?? join(HERE, "corpus.json");
  const base = JSON.parse(readFileSync(corpusPath, "utf8"));
  const corpus = expandCorpus(base, TARGET_DOCS);

  console.log(`▸ embedder : ${cfg.embedder.model} @ ${cfg.embedder.baseUrl}`);
  console.log(`▸ vecvec   : ${cfg.vecvec.collection} @ ${cfg.vecvec.url} (${cfg.metric})`);
  console.log(`▸ corpus   : ${corpus.length} documents (${base.length} base × variants, ${corpusPath})\n`);

  // Fail fast if the server is down, with a helpful hint.
  await vvHealth(cfg);

  // Embed everything. The first vector tells us the collection dimension.
  const texts = corpus.map((d) => d.text);
  const t0 = Date.now();
  process.stdout.write("embedding documents… ");
  const vectors = await embed(cfg, texts, {
    onProgress: (done, total) => {
      process.stdout.write(`\rembedding documents… ${done}/${total}`);
    },
  });
  const dim = vectors[0].length;
  console.log(`\r✓ embedded ${vectors.length} documents · ${dim}d · ${Date.now() - t0}ms`);

  // Recreate the collection to match this model's dimension, then upsert.
  await recreateCollection(cfg, dim);
  console.log(`✓ collection "${cfg.vecvec.collection}" created (${dim}d, ${cfg.metric})`);

  // One commit per vector: upsert a single point, then commit, so the collection
  // timeline gets a distinct version for every document added.
  let lastVersion;
  for (let i = 0; i < corpus.length; i++) {
    const doc = corpus[i];
    const point = {
      vector: prepVector(cfg, vectors[i]),
      payload: { text: doc.text, category: doc.category ?? null },
    };
    await upsertPoints(cfg, [point]);

    const snippet = doc.text.length > 44 ? `${doc.text.slice(0, 44)}…` : doc.text;
    const tag = i === 0 ? "genesis" : i === corpus.length - 1 ? "seed" : undefined;
    const c = await commit(cfg, `add #${i + 1}: ${snippet}`, tag);
    lastVersion = c.version;
    process.stdout.write(`\rcommitting one per vector… ${i + 1}/${corpus.length} (v${c.version})`);
  }
  console.log(`\r✓ ${corpus.length} vectors, ${corpus.length} commits (head v${lastVersion})        `);

  const stats = await collectionStats(cfg);
  console.log(`\n✓ done — ${stats.count} points live. Now run:  node query.mjs "your question"`);
}

main().catch((e) => {
  console.error(`\n✗ seed failed: ${e.message}\n`);
  process.exit(1);
});
