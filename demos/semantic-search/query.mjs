#!/usr/bin/env node
// Semantic search against the seeded vecvec collection.
//
//   node query.mjs "how does a database survive a crash?"   # one-shot
//   node query.mjs                                          # interactive REPL
//   TOP_K=8 node query.mjs "..."                            # more results
//
// Embeds the query with the same Ollama model used at seed time, runs an HNSW
// nearest-neighbour search, then fetches each hit's stored text to display it.

import { createInterface } from "node:readline";

import {
  loadConfig,
  embedOne,
  prepVector,
  vvHealth,
  queryVector,
  getPoint,
  collectionStats,
} from "./lib.mjs";

async function runQuery(cfg, text) {
  const t0 = Date.now();
  const vector = prepVector(cfg, await embedOne(cfg, text));
  const embedMs = Date.now() - t0;

  const t1 = Date.now();
  const { results } = await queryVector(cfg, vector, cfg.topK);
  const searchMs = Date.now() - t1;

  if (!results.length) {
    console.log("  (no results — did you run `node seed.mjs` first?)\n");
    return;
  }

  // query returns {id, score}; fetch each point to show its text/category.
  const hits = await Promise.all(
    results.map(async ({ id, score }) => {
      const p = await getPoint(cfg, id).catch(() => null);
      return { id, score, payload: p?.payload ?? {} };
    }),
  );

  console.log(`\n  “${text}”   (embed ${embedMs}ms · search ${searchMs}ms)\n`);
  hits.forEach(({ id, score, payload }, i) => {
    const cat = payload.category ? `  [${payload.category}]` : "";
    console.log(`  ${String(i + 1).padStart(2)}. ${score.toFixed(4)}${cat}`);
    console.log(`      ${payload.text ?? `(point #${id} — no text payload)`}`);
  });
  console.log();
}

async function main() {
  const cfg = loadConfig();
  await vvHealth(cfg);

  // Verify the collection exists so we can give a clear hint instead of a 404.
  try {
    const stats = await collectionStats(cfg);
    if (stats.count === 0) {
      console.error(`\n✗ collection "${cfg.vecvec.collection}" is empty. Seed it first:  node seed.mjs\n`);
      process.exit(1);
    }
  } catch {
    console.error(`\n✗ collection "${cfg.vecvec.collection}" not found. Seed it first:  node seed.mjs\n`);
    process.exit(1);
  }

  const arg = process.argv.slice(2).join(" ").trim();
  if (arg) {
    await runQuery(cfg, arg);
    return;
  }

  // Interactive REPL. Iterate lines serially with `for await` so each query fully
  // completes before the next line is read (an event handler with `await` races on
  // piped/pasted input and can exit before a query prints).
  console.log(`semantic search · ${cfg.vecvec.collection} · model ${cfg.embedder.model}`);
  console.log("type a query (or Ctrl-D / 'exit' to quit)\n");
  const rl = createInterface({ input: process.stdin, output: process.stdout });
  process.stdout.write("› ");
  for await (const line of rl) {
    const q = line.trim();
    if (q === "exit" || q === "quit") break;
    if (q) {
      try {
        await runQuery(cfg, q);
      } catch (e) {
        console.error(`  ✗ ${e.message}\n`);
      }
    }
    process.stdout.write("› ");
  }
  rl.close();
  console.log("\nbye.");
}

main().catch((e) => {
  console.error(`\n✗ query failed: ${e.message}\n`);
  process.exit(1);
});
