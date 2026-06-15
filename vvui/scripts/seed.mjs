#!/usr/bin/env node
// Seed a vecvec server with demo collections so the SCOPE has something to explore:
// clustered vectors with rich payloads, committed across several versions (with a
// delete) so the timeline + diffs are meaningful.
//
//   node scripts/seed.mjs            # targets http://127.0.0.1:6333
//   VECVEC_REST_URL=… node scripts/seed.mjs

const BASE = process.env.VECVEC_REST_URL ?? "http://127.0.0.1:6333";

async function api(method, path, body) {
  const res = await fetch(BASE + path, {
    method,
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  const text = await res.text();
  let json = null;
  try {
    json = text ? JSON.parse(text) : null;
  } catch {
    json = text; // e.g. /healthz returns plain "ok"
  }
  if (!res.ok) throw new Error(`${method} ${path} → ${res.status}: ${json?.error ?? text}`);
  return json;
}

// deterministic-ish RNG so reruns look similar
let seed = 1337;
function rnd() {
  seed = (seed * 1664525 + 1013904223) >>> 0;
  return seed / 0xffffffff;
}
function gauss() {
  return Math.sqrt(-2 * Math.log(rnd() + 1e-9)) * Math.cos(2 * Math.PI * rnd());
}
function pick(arr) {
  return arr[Math.floor(rnd() * arr.length)];
}

function clusterCenters(k, dim) {
  return Array.from({ length: k }, () => Array.from({ length: dim }, () => gauss()));
}
function near(center, spread) {
  return center.map((c) => Math.round((c + gauss() * spread) * 1000) / 1000);
}

async function ensureFresh(name, dim, metric) {
  try {
    await api("DELETE", `/collections/${name}`);
  } catch {
    /* not present — fine */
  }
  await api("POST", `/collections/${name}`, { dim, metric });
}

async function seedConcepts() {
  const name = "concepts-24d";
  const dim = 24;
  const topics = ["physics", "music", "cuisine", "finance", "sport"];
  const langs = ["en", "es", "de", "fr", "ja", "zh", "pt", "it", "ru"];
  const centers = clusterCenters(topics.length, dim);

  await ensureFresh(name, dim, "cosine");
  console.log(`\n▸ ${name} · ${dim}d · cosine`);

  const makeBatch = (n, yearBase) =>
    Array.from({ length: n }, () => {
      const t = Math.floor(rnd() * topics.length);
      return {
        vector: near(centers[t], 0.35),
        payload: {
          topic: topics[t],
          lang: pick(langs),
          year: yearBase + Math.floor(rnd() * 4),
          popularity: Math.round(rnd() * 100),
        },
      };
    });

  let r = await api("POST", `/collections/${name}/points`, { points: makeBatch(150, 2018) });
  console.log(`  + ${r.inserted} points (genesis)`);
  await api("POST", `/collections/${name}/commit`, { message: "genesis — 5 topic clusters", tag: "genesis" });
  console.log("  ◈ commit genesis");

  r = await api("POST", `/collections/${name}/points`, { points: makeBatch(90, 2021) });
  console.log(`  + ${r.inserted} points (expansion)`);
  // delete a handful from the first batch to make the diff interesting
  for (const id of [3, 11, 27, 42, 88]) {
    await api("DELETE", `/collections/${name}/points/${id}`).catch(() => {});
  }
  console.log("  − 5 points tombstoned");
  await api("POST", `/collections/${name}/commit`, { message: "expansion + prune" });
  console.log("  ◈ commit expansion");

  r = await api("POST", `/collections/${name}/points`, { points: makeBatch(70, 2023) });
  console.log(`  + ${r.inserted} points (refresh)`);
  await api("POST", `/collections/${name}/commit`, { message: "2023 refresh", tag: "v1.0" });
  console.log("  ◈ commit refresh @v1.0");
}

async function seedPixels() {
  const name = "pixels-16d";
  const dim = 16;
  const kinds = ["cat", "dog", "car", "tree"];
  const centers = clusterCenters(kinds.length, dim);

  await ensureFresh(name, dim, "dot");
  console.log(`\n▸ ${name} · ${dim}d · dot`);

  const batch = (n) =>
    Array.from({ length: n }, () => {
      const t = Math.floor(rnd() * kinds.length);
      return {
        vector: near(centers[t], 0.5),
        payload: { kind: kinds[t], brightness: Math.round(rnd() * 100) / 100 },
      };
    });

  let r = await api("POST", `/collections/${name}/points`, { points: batch(120) });
  console.log(`  + ${r.inserted} points`);
  await api("POST", `/collections/${name}/commit`, { message: "initial import", tag: "base" });
  console.log("  ◈ commit base");

  r = await api("POST", `/collections/${name}/points`, { points: batch(60) });
  console.log(`  + ${r.inserted} points`);
  await api("POST", `/collections/${name}/commit`, { message: "more samples" });
  console.log("  ◈ commit more samples");
}

async function seedStream() {
  // A long-lived collection that grows on every commit, so the timeline has lots
  // of versions. Total point count scales linearly from 1 to ~10000 across 100
  // commits, so every commit adds roughly the same number of points (~100).
  const name = "stream-32d";
  const dim = 32;
  const sources = ["sensor", "log", "trace", "metric", "event"];
  const centers = clusterCenters(sources.length, dim);

  await ensureFresh(name, dim, "cosine");
  console.log(`\n▸ ${name} · ${dim}d · cosine`);

  const batch = (n, hour) =>
    Array.from({ length: n }, () => {
      const t = Math.floor(rnd() * sources.length);
      return {
        vector: near(centers[t], 0.4),
        payload: {
          source: sources[t],
          hour,
          score: Math.round(rnd() * 1000) / 1000,
        },
      };
    });

  const COMMITS = 100;
  const TARGET = 10000;
  // Linear schedule: cumulative count at commit i is TARGET * i / COMMITS,
  // so each commit adds roughly the same number of points.
  let total = 0;
  for (let i = 1; i <= COMMITS; i++) {
    const want = Math.round((TARGET * i) / COMMITS);
    const n = Math.max(1, want - total);

    const r = await api("POST", `/collections/${name}/points`, { points: batch(n, i) });
    total += r.inserted;

    // occasionally tombstone a few older points to keep diffs interesting
    let pruned = 0;
    if (i % 10 === 0) {
      for (let k = 0; k < 3; k++) {
        const id = Math.floor(rnd() * total);
        await api("DELETE", `/collections/${name}/points/${id}`)
          .then(() => pruned++)
          .catch(() => {});
      }
    }

    const tag = i === 1 ? "genesis" : i % 25 === 0 ? `h${i}` : undefined;
    const msg = pruned
      ? `commit ${i}: +${r.inserted}, −${pruned} (total ${total})`
      : `commit ${i}: +${r.inserted} (total ${total})`;
    await api("POST", `/collections/${name}/commit`, { message: msg, tag });
    console.log(`  ◈ ${msg}${tag ? ` @${tag}` : ""}`);
  }
}

async function main() {
  console.log(`seeding vecvec @ ${BASE}`);
  try {
    await api("GET", "/healthz");
  } catch {
    console.error(`\n✗ cannot reach vecvec at ${BASE}\n  start it with:  cargo run -p vecvec-server\n`);
    process.exit(1);
  }
  await seedConcepts();
  await seedPixels();
  await seedStream();
  console.log("\n✓ done — open the SCOPE and explore.\n");
}

main().catch((e) => {
  console.error("\n✗ seed failed:", e.message, "\n");
  process.exit(1);
});
