import { create } from "zustand";
import { api, ApiError } from "./api";
import type {
  CollectionStats,
  Filter,
  Metric,
  Payload,
  PointRecord,
  ProjectionKind,
  ScoredPoint,
  VersionInfo,
} from "./types";
import { PROJECTION_CAP } from "./lib/projection";
import { bestColorKey } from "./lib/format";

export interface Toast {
  id: number;
  kind: "ok" | "err" | "info";
  title: string;
  msg: string;
}

export interface QueryHighlight {
  origin: number | null; // point id the query radiated from, if any
  label: string;
  results: ScoredPoint[];
  ids: Set<number>;
}

export interface ViewDiff {
  added: Set<number>;
  removed: number[];
}

interface State {
  // connection
  connected: boolean | null;

  // collections
  collections: CollectionStats[];
  active: string | null;
  stats: CollectionStats | null;

  // points (current view, possibly at a past version)
  points: PointRecord[];
  total: number;
  truncated: boolean;

  // versions / timeline
  versions: VersionInfo[];
  head: number | null;
  viewVersion: number | null; // null = live HEAD
  diff: ViewDiff | null;

  // scope controls
  projection: ProjectionKind;
  colorKey: string | null;

  // selection + query
  selectedPoint: number | null;
  highlight: QueryHighlight | null;

  // status
  loadingPoints: boolean;

  toasts: Toast[];

  // actions
  init: () => Promise<void>;
  refreshCollections: () => Promise<void>;
  selectCollection: (name: string | null) => Promise<void>;
  reloadView: () => Promise<void>;
  setViewVersion: (v: number | null) => Promise<void>;
  setProjection: (p: ProjectionKind) => void;
  setColorKey: (k: string | null) => void;
  selectPoint: (id: number | null) => void;

  createCollection: (name: string, dim: number, metric: Metric) => Promise<boolean>;
  dropCollection: (name: string) => Promise<void>;
  upsertPoints: (points: { vector: number[]; payload?: Payload }[]) => Promise<number[] | null>;
  deletePoint: (id: number) => Promise<void>;
  commit: (message?: string, tag?: string) => Promise<void>;
  tagVersion: (tag: string, version: number) => Promise<void>;
  branchVersion: (branch: string, version: number) => Promise<void>;
  restoreVersion: (version: number) => Promise<void>;

  runQuery: (vector: number[], k: number, filter?: Filter, label?: string) => Promise<void>;
  runRecommend: (
    positive: number[],
    negative: number[],
    k: number,
    filter?: Filter,
  ) => Promise<void>;
  queryByPoint: (id: number, k: number) => Promise<void>;
  clearHighlight: () => void;

  toast: (kind: Toast["kind"], title: string, msg: string) => void;
  dismissToast: (id: number) => void;
}

let toastSeq = 1;

export const useStore = create<State>((set, get) => ({
  connected: null,
  collections: [],
  active: null,
  stats: null,
  points: [],
  total: 0,
  truncated: false,
  versions: [],
  head: null,
  viewVersion: null,
  diff: null,
  projection: "pca",
  colorKey: null,
  selectedPoint: null,
  highlight: null,
  loadingPoints: false,
  toasts: [],

  toast: (kind, title, msg) => {
    const id = toastSeq++;
    set((s) => ({ toasts: [...s.toasts, { id, kind, title, msg }] }));
    setTimeout(() => get().dismissToast(id), kind === "err" ? 6000 : 3500);
  },
  dismissToast: (id) => set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) })),

  init: async () => {
    try {
      await api.health();
      set({ connected: true });
      await get().refreshCollections();
      const first = get().collections[0]?.name ?? null;
      if (first) await get().selectCollection(first);
    } catch {
      set({ connected: false });
    }
  },

  refreshCollections: async () => {
    try {
      const collections = await api.listCollections();
      set({ connected: true, collections });
    } catch (e) {
      set({ connected: false });
      throw e;
    }
  },

  selectCollection: async (name) => {
    if (name === null) {
      set({ active: null, stats: null, points: [], versions: [], head: null });
      return;
    }
    set({
      active: name,
      viewVersion: null,
      selectedPoint: null,
      highlight: null,
      diff: null,
      colorKey: null,
      points: [],
    });
    await get().reloadView();
  },

  reloadView: async () => {
    const name = get().active;
    if (!name) return;
    set({ loadingPoints: true });
    try {
      const [stats, vlist] = await Promise.all([
        api.collectionStats(name),
        api.listVersions(name),
      ]);
      const viewVersion = get().viewVersion;
      const at = viewVersion != null ? { version: viewVersion } : undefined;
      const { points, total } = await api.scroll(name, { limit: PROJECTION_CAP, at });

      // Default the color key to the most categorical payload field.
      let colorKey = get().colorKey;
      if (colorKey === null) colorKey = bestColorKey(points);

      // Compute the diff this version introduced (vs its parent), for the readout.
      let diff: ViewDiff | null = null;
      const effective = viewVersion ?? vlist.head;
      if (effective != null) {
        const v = vlist.versions.find((x) => x.version === effective);
        if (v && v.parent != null) {
          try {
            const d = await api.diff(name, v.parent, effective);
            diff = { added: new Set(d.added), removed: d.removed };
          } catch {
            diff = null;
          }
        }
      }

      set({
        stats,
        versions: vlist.versions,
        head: vlist.head,
        points,
        total,
        truncated: total > points.length,
        colorKey,
        diff,
        loadingPoints: false,
      });
      // keep collection list counts fresh
      get().refreshCollections().catch(() => {});
    } catch (e) {
      set({ loadingPoints: false });
      get().toast("err", "load failed", errMsg(e));
    }
  },

  setViewVersion: async (v) => {
    set({ viewVersion: v, highlight: null });
    await get().reloadView();
  },

  setProjection: (p) => set({ projection: p }),
  setColorKey: (k) => set({ colorKey: k }),
  selectPoint: (id) => set({ selectedPoint: id }),

  createCollection: async (name, dim, metric) => {
    try {
      await api.createCollection(name, dim, metric);
      await get().refreshCollections();
      await get().selectCollection(name);
      get().toast("ok", "collection created", `${name} · ${dim}d · ${metric}`);
      return true;
    } catch (e) {
      get().toast("err", "create failed", errMsg(e));
      return false;
    }
  },

  dropCollection: async (name) => {
    try {
      await api.dropCollection(name);
      const wasActive = get().active === name;
      await get().refreshCollections();
      if (wasActive) {
        const next = get().collections[0]?.name ?? null;
        await get().selectCollection(next);
      }
      get().toast("ok", "collection dropped", name);
    } catch (e) {
      get().toast("err", "drop failed", errMsg(e));
    }
  },

  upsertPoints: async (points) => {
    const name = get().active;
    if (!name) return null;
    try {
      const res = await api.upsert(name, points);
      get().toast("ok", "points inserted", `+${res.inserted} → id ${res.ids.join(", ")}`);
      set({ viewVersion: null });
      await get().reloadView();
      return res.ids;
    } catch (e) {
      get().toast("err", "insert failed", errMsg(e));
      return null;
    }
  },

  deletePoint: async (id) => {
    const name = get().active;
    if (!name) return;
    try {
      const res = await api.deletePoint(name, id);
      if (res.deleted) get().toast("ok", "point deleted", `id ${id} tombstoned`);
      else get().toast("info", "no-op", `id ${id} was already gone`);
      if (get().selectedPoint === id) set({ selectedPoint: null });
      set({ viewVersion: null });
      await get().reloadView();
    } catch (e) {
      get().toast("err", "delete failed", errMsg(e));
    }
  },

  commit: async (message, tag) => {
    const name = get().active;
    if (!name) return;
    try {
      const { version } = await api.commit(name, { message, tag });
      get().toast("ok", "committed", `v${version}${tag ? ` · @${tag}` : ""}`);
      set({ viewVersion: null });
      await get().reloadView();
    } catch (e) {
      get().toast("err", "commit failed", errMsg(e));
    }
  },

  tagVersion: async (tag, version) => {
    const name = get().active;
    if (!name) return;
    try {
      await api.createTag(name, tag, version);
      get().toast("ok", "tag set", `@${tag} → v${version}`);
      await get().reloadView();
    } catch (e) {
      get().toast("err", "tag failed", errMsg(e));
    }
  },

  branchVersion: async (branch, version) => {
    const name = get().active;
    if (!name) return;
    try {
      await api.createBranch(name, branch, version);
      get().toast("ok", "branch set", `${branch} → v${version}`);
      await get().reloadView();
    } catch (e) {
      get().toast("err", "branch failed", errMsg(e));
    }
  },

  restoreVersion: async (version) => {
    const name = get().active;
    if (!name) return;
    try {
      const res = await api.restore(name, version);
      get().toast("ok", "restored", `v${version} → new v${res.version}`);
      set({ viewVersion: null });
      await get().reloadView();
    } catch (e) {
      get().toast("err", "restore failed", errMsg(e));
    }
  },

  runQuery: async (vector, k, filter, label) => {
    const name = get().active;
    if (!name) return;
    try {
      const at = get().viewVersion != null ? { version: get().viewVersion! } : undefined;
      const results = await api.query(name, { vector, k, at, filter });
      set({
        highlight: {
          origin: null,
          label: label ?? "vector query",
          results,
          ids: new Set(results.map((r) => r.id)),
        },
      });
      get().toast("ok", "query", `${results.length} matches`);
    } catch (e) {
      get().toast("err", "query failed", errMsg(e));
    }
  },

  runRecommend: async (positive, negative, k, filter) => {
    const name = get().active;
    if (!name) return;
    try {
      const results = await api.recommend(name, { positive, negative, k, filter });
      set({
        highlight: {
          origin: positive[0] ?? null,
          label: `recommend +${positive.length}/−${negative.length}`,
          results,
          ids: new Set(results.map((r) => r.id)),
        },
      });
      get().toast("ok", "recommend", `${results.length} matches`);
    } catch (e) {
      get().toast("err", "recommend failed", errMsg(e));
    }
  },

  queryByPoint: async (id, k) => {
    const name = get().active;
    if (!name) return;
    const pt = get().points.find((p) => p.id === id);
    try {
      let vector = pt?.vector;
      if (!vector) vector = (await api.getPoint(name, id)).vector;
      const at = get().viewVersion != null ? { version: get().viewVersion! } : undefined;
      const results = (await api.query(name, { vector, k: k + 1, at })).filter(
        (r) => r.id !== id,
      );
      set({
        highlight: {
          origin: id,
          label: `neighbors of #${id}`,
          results: results.slice(0, k),
          ids: new Set(results.slice(0, k).map((r) => r.id)),
        },
        selectedPoint: id,
      });
    } catch (e) {
      get().toast("err", "query failed", errMsg(e));
    }
  },

  clearHighlight: () => set({ highlight: null }),
}));

function errMsg(e: unknown): string {
  if (e instanceof ApiError) return e.message;
  if (e instanceof Error) return e.message;
  return String(e);
}
