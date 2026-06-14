export type Metric = "cosine" | "dot" | "euclidean";

export interface CollectionStats {
  name: string;
  dim: number;
  metric: Metric;
  count: number;
  head_version: number | null;
}

export type Payload = Record<string, unknown> | null;

export interface PointRecord {
  id: number;
  vector: number[];
  payload: Payload;
}

export interface ScoredPoint {
  id: number;
  score: number;
}

export interface VersionInfo {
  version: number;
  parent: number | null;
  created_at_ms: number;
  trigger: string;
  message: string | null;
}

export interface VersionList {
  versions: VersionInfo[];
  head: number | null;
}

export interface DiffResult {
  added: number[];
  removed: number[];
}

/** A version selector understood by the scroll / query "at" parameter. */
export interface At {
  version?: number;
  tag?: string;
  branch?: string;
}

export type ProjectionKind = "pca" | "umap";

/** A point as laid out in the 2D scope, with its source data attached. */
export interface ProjectedPoint {
  id: number;
  x: number; // normalized 0..1
  y: number; // normalized 0..1
  payload: Payload;
}

/** Qdrant-style filter condition (subset the server supports). */
export interface FilterCondition {
  key: string;
  match?: unknown;
  range?: { gt?: number; gte?: number; lt?: number; lte?: number };
}

export interface Filter {
  must?: FilterCondition[];
  should?: FilterCondition[];
  must_not?: FilterCondition[];
}
