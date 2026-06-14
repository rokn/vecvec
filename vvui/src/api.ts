import type {
  At,
  CollectionStats,
  DiffResult,
  Filter,
  Metric,
  Payload,
  PointRecord,
  ScoredPoint,
  VersionList,
} from "./types";

const BASE = "/api";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
    this.name = "ApiError";
  }
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(BASE + path, {
      headers: { "Content-Type": "application/json" },
      ...init,
    });
  } catch {
    throw new ApiError(0, "no connection to vecvec server");
  }
  const text = await res.text();
  const body = text ? safeParse(text) : null;
  if (!res.ok) {
    const msg =
      (body && typeof body === "object" && "error" in body
        ? String((body as { error: unknown }).error)
        : null) ?? `${res.status} ${res.statusText}`;
    throw new ApiError(res.status, msg);
  }
  return body as T;
}

function safeParse(t: string): unknown {
  try {
    return JSON.parse(t);
  } catch {
    return t;
  }
}

function atQuery(at?: At): string {
  if (!at) return "";
  if (at.version != null) return `&version=${at.version}`;
  if (at.tag) return `&tag=${encodeURIComponent(at.tag)}`;
  if (at.branch) return `&branch=${encodeURIComponent(at.branch)}`;
  return "";
}

export const api = {
  health: () => req<string>("/healthz"),

  listCollections: () =>
    req<{ collections: CollectionStats[] }>("/collections").then((r) => r.collections),

  collectionStats: (name: string) =>
    req<CollectionStats>(`/collections/${encodeURIComponent(name)}`),

  createCollection: (name: string, dim: number, metric: Metric) =>
    req<{ name: string }>(`/collections/${encodeURIComponent(name)}`, {
      method: "POST",
      body: JSON.stringify({ dim, metric }),
    }),

  dropCollection: (name: string) =>
    req<{ dropped: string }>(`/collections/${encodeURIComponent(name)}`, {
      method: "DELETE",
    }),

  upsert: (name: string, points: { vector: number[]; payload?: Payload }[]) =>
    req<{ inserted: number; ids: number[] }>(
      `/collections/${encodeURIComponent(name)}/points`,
      { method: "POST", body: JSON.stringify({ points }) },
    ),

  scroll: (
    name: string,
    opts: { offset?: number; limit?: number; at?: At } = {},
  ) =>
    req<{ points: PointRecord[]; total: number }>(
      `/collections/${encodeURIComponent(name)}/points?offset=${
        opts.offset ?? 0
      }&limit=${opts.limit ?? 2000}${atQuery(opts.at)}`,
    ),

  getPoint: (name: string, id: number) =>
    req<PointRecord>(`/collections/${encodeURIComponent(name)}/points/${id}`),

  deletePoint: (name: string, id: number) =>
    req<{ deleted: boolean; id: number }>(
      `/collections/${encodeURIComponent(name)}/points/${id}`,
      { method: "DELETE" },
    ),

  query: (
    name: string,
    opts: { vector: number[]; k: number; at?: At; filter?: Filter },
  ) =>
    req<{ results: ScoredPoint[] }>(
      `/collections/${encodeURIComponent(name)}/query`,
      { method: "POST", body: JSON.stringify(opts) },
    ).then((r) => r.results),

  recommend: (
    name: string,
    opts: { positive: number[]; negative: number[]; k: number; filter?: Filter },
  ) =>
    req<{ results: ScoredPoint[] }>(
      `/collections/${encodeURIComponent(name)}/recommend`,
      { method: "POST", body: JSON.stringify(opts) },
    ).then((r) => r.results),

  commit: (name: string, opts: { message?: string; tag?: string } = {}) =>
    req<{ version: number }>(`/collections/${encodeURIComponent(name)}/commit`, {
      method: "POST",
      body: JSON.stringify(opts),
    }),

  listVersions: (name: string) =>
    req<VersionList>(`/collections/${encodeURIComponent(name)}/versions`),

  diff: (name: string, from: number, to: number) =>
    req<DiffResult>(
      `/collections/${encodeURIComponent(name)}/diff?from=${from}&to=${to}`,
    ),

  createTag: (name: string, tag: string, version: number) =>
    req<unknown>(`/collections/${encodeURIComponent(name)}/tags`, {
      method: "POST",
      body: JSON.stringify({ name: tag, version }),
    }),

  createBranch: (name: string, branch: string, version: number) =>
    req<unknown>(`/collections/${encodeURIComponent(name)}/branches`, {
      method: "POST",
      body: JSON.stringify({ name: branch, version }),
    }),

  restore: (name: string, version: number) =>
    req<{ version: number }>(
      `/collections/${encodeURIComponent(name)}/restore`,
      { method: "POST", body: JSON.stringify({ version }) },
    ),
};
