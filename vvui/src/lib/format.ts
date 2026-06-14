import type { Payload, PointRecord } from "../types";

export function fmtNum(n: number): string {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1).replace(/\.0$/, "") + "M";
  if (n >= 10_000) return (n / 1000).toFixed(1).replace(/\.0$/, "") + "k";
  return n.toLocaleString();
}

export function relTime(ms: number): string {
  const d = Date.now() - ms;
  const s = Math.round(d / 1000);
  if (s < 5) return "just now";
  if (s < 60) return `${s}s ago`;
  const m = Math.round(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.round(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.round(h / 24)}d ago`;
}

export function clockTime(ms: number): string {
  const d = new Date(ms);
  const p = (n: number) => String(n).padStart(2, "0");
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}

export function vecPreview(v: number[], n = 4): string {
  const head = v
    .slice(0, n)
    .map((x) => (x >= 0 ? " " : "") + x.toFixed(3))
    .join(", ");
  return `[${head}${v.length > n ? ", …" : ""}]`;
}

/** All distinct top-level payload keys across a set of points. */
export function payloadKeys(points: PointRecord[]): string[] {
  const keys = new Set<string>();
  for (const p of points) {
    if (p.payload && typeof p.payload === "object") {
      for (const k of Object.keys(p.payload)) keys.add(k);
    }
  }
  return Array.from(keys).sort();
}

export function payloadValue(payload: Payload, key: string): unknown {
  if (!payload || typeof payload !== "object") return undefined;
  return (payload as Record<string, unknown>)[key];
}

/** Pick the payload field that will color the scope most legibly: the most
 *  "categorical" one (few distinct values, ideally string-valued). */
export function bestColorKey(points: PointRecord[]): string | null {
  const keys = payloadKeys(points);
  if (keys.length === 0) return null;
  let best: string | null = null;
  let bestScore = Infinity;
  for (const k of keys) {
    const distinct = new Set<string>();
    let strings = 0;
    let seen = 0;
    for (const p of points) {
      const v = payloadValue(p.payload, k);
      if (v === undefined) continue;
      seen++;
      if (typeof v === "string") strings++;
      distinct.add(fmtVal(v));
      if (distinct.size > 40) break;
    }
    if (seen === 0 || distinct.size < 2) continue;
    // prefer 2..14 distinct values; string-valued fields get a bonus
    const inSweet = distinct.size >= 2 && distinct.size <= 14;
    const score = distinct.size + (inSweet ? 0 : 50) + (strings > seen / 2 ? 0 : 6);
    if (score < bestScore) {
      bestScore = score;
      best = k;
    }
  }
  return best ?? keys[0];
}

/** Categorical palette tuned for a near-black phosphor instrument. */
export const PALETTE = [
  "#4af2c8", // mint (primary phosphor)
  "#7fb4ff", // sky
  "#9be23c", // lime
  "#ffb162", // amber
  "#fb7b8e", // rose
  "#c79bff", // violet
  "#5fe0e6", // cyan
  "#ffd24a", // gold
  "#6effa6", // spring
  "#ff9bd0", // pink
  "#a0a8ff", // periwinkle
  "#e0e6ad", // sand
];

const numberFmt = new Intl.NumberFormat("en-US", { maximumFractionDigits: 3 });

export function fmtScore(s: number): string {
  return numberFmt.format(s);
}

export function payloadSummary(payload: Payload): string {
  if (!payload || typeof payload !== "object") return "—";
  const entries = Object.entries(payload as Record<string, unknown>);
  if (entries.length === 0) return "—";
  return entries
    .slice(0, 4)
    .map(([k, v]) => `${k}=${fmtVal(v)}`)
    .join("  ");
}

export function fmtVal(v: unknown): string {
  if (v === null || v === undefined) return "∅";
  if (typeof v === "number") return numberFmt.format(v);
  if (typeof v === "string") return v;
  if (typeof v === "boolean") return v ? "true" : "false";
  return JSON.stringify(v);
}
