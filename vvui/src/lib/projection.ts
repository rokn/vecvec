import { UMAP } from "umap-js";
import type { ProjectionKind } from "../types";
import { pca2d } from "./pca";

/** Cap on how many points we project at once (UMAP especially is superlinear). */
export const PROJECTION_CAP = 2500;

function lcg(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0xffffffff;
  };
}

function umap2d(vectors: number[][]): [number, number][] {
  const n = vectors.length;
  const umap = new UMAP({
    nComponents: 2,
    nNeighbors: Math.max(2, Math.min(15, n - 1)),
    minDist: 0.12,
    spread: 1.0,
    random: lcg(0xc0ffee),
  });
  const coords = umap.fit(vectors);
  return coords.map((c) => [c[0], c[1]] as [number, number]);
}

/** Fit a 2D layout into the unit square, preserving aspect ratio and centering. */
function normalize(raw: [number, number][]): [number, number][] {
  if (raw.length === 0) return [];
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const [x, y] of raw) {
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  const spanX = maxX - minX || 1;
  const spanY = maxY - minY || 1;
  const span = Math.max(spanX, spanY);
  const pad = 0.06;
  const usable = 1 - pad * 2;
  const offX = (span - spanX) / 2;
  const offY = (span - spanY) / 2;
  return raw.map(([x, y]) => [
    pad + ((x - minX + offX) / span) * usable,
    pad + ((y - minY + offY) / span) * usable,
  ]);
}

export function project(
  kind: ProjectionKind,
  vectors: number[][],
): [number, number][] {
  if (vectors.length === 0) return [];
  if (vectors.length < 4) {
    // Too few points for a meaningful layout — fan them out deterministically.
    return normalize(
      vectors.map((_, i) => {
        const a = (i / vectors.length) * Math.PI * 2;
        return [Math.cos(a), Math.sin(a)] as [number, number];
      }),
    );
  }
  const raw = kind === "umap" && vectors.length >= 10 ? umap2d(vectors) : pca2d(vectors);
  return normalize(raw);
}
