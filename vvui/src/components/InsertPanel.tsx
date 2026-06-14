import { useState } from "react";
import { useStore } from "../store";
import type { Payload } from "../types";

function randomVec(dim: number): number[] {
  return Array.from({ length: dim }, () => Math.round((Math.random() * 2 - 1) * 1000) / 1000);
}

/** A few clustered "blobs" so a fresh collection projects into visible structure. */
function clusteredVec(dim: number, cluster: number, clusters: number): number[] {
  const center = (i: number) =>
    Math.sin((cluster + 1) * 1.7 + i * (0.3 + cluster / clusters)) * 0.8;
  return Array.from(
    { length: dim },
    (_, i) => Math.round((center(i) + (Math.random() * 2 - 1) * 0.28) * 1000) / 1000,
  );
}

export function InsertPanel() {
  const stats = useStore((s) => s.stats);
  const upsert = useStore((s) => s.upsertPoints);
  const toast = useStore((s) => s.toast);

  const dim = stats?.dim ?? 0;
  const [mode, setMode] = useState<"single" | "batch">("single");
  const [vec, setVec] = useState("");
  const [payload, setPayload] = useState('{\n  "group": "a"\n}');
  const [count, setCount] = useState(60);
  const [clusters, setClusters] = useState(4);
  const [busy, setBusy] = useState(false);

  if (!stats) return null;

  const parsePayload = (): Payload | undefined => {
    const t = payload.trim();
    if (!t || t === "{}") return undefined;
    try {
      const v = JSON.parse(t);
      return typeof v === "object" && v !== null ? v : undefined;
    } catch {
      toast("err", "bad payload", "payload must be valid JSON object");
      throw new Error("bad payload");
    }
  };

  const submitSingle = async () => {
    const nums = vec
      .split(/[\s,]+/)
      .map((x) => x.trim())
      .filter(Boolean)
      .map(Number);
    if (nums.length !== dim || nums.some(Number.isNaN)) {
      toast("err", "bad vector", `need ${dim} numbers, got ${nums.length}`);
      return;
    }
    let p: Payload | undefined;
    try {
      p = parsePayload();
    } catch {
      return;
    }
    setBusy(true);
    await upsert([{ vector: nums, payload: p }]);
    setBusy(false);
    setVec("");
  };

  const submitBatch = async () => {
    let template: Payload | undefined;
    try {
      template = parsePayload();
    } catch {
      return;
    }
    const n = Math.max(1, Math.min(2000, count | 0));
    const k = Math.max(1, Math.min(12, clusters | 0));
    const points = Array.from({ length: n }, (_, i) => {
      const c = i % k;
      const base = template && typeof template === "object" ? { ...template } : {};
      return {
        vector: clusteredVec(dim, c, k),
        payload: { ...base, cluster: c } as Payload,
      };
    });
    setBusy(true);
    await upsert(points);
    setBusy(false);
  };

  return (
    <div>
      <div className="subhead">create points</div>

      <div className="seg" style={{ marginBottom: 14, width: "100%" }}>
        <button
          className={mode === "single" ? "on" : ""}
          style={{ flex: 1 }}
          onClick={() => setMode("single")}
        >
          single
        </button>
        <button
          className={mode === "batch" ? "on" : ""}
          style={{ flex: 1 }}
          onClick={() => setMode("batch")}
        >
          random batch
        </button>
      </div>

      {mode === "single" ? (
        <>
          <div className="field">
            <label>
              vector · {dim} dims
              <button
                className="btn ghost sm"
                style={{ float: "right", padding: "0 4px" }}
                onClick={() => setVec(randomVec(dim).join(", "))}
              >
                ↯ random
              </button>
            </label>
            <textarea
              className="textarea"
              placeholder={`${dim} comma/space-separated floats`}
              value={vec}
              onChange={(e) => setVec(e.target.value)}
              style={{ minHeight: 90, fontSize: 11 }}
            />
          </div>
          <div className="field">
            <label>payload · json</label>
            <textarea
              className="textarea"
              value={payload}
              onChange={(e) => setPayload(e.target.value)}
              style={{ fontSize: 11 }}
            />
          </div>
          <button className="btn primary" style={{ width: "100%" }} disabled={busy} onClick={submitSingle}>
            {busy ? "inserting…" : "◢ insert point"}
          </button>
        </>
      ) : (
        <>
          <p className="muted" style={{ fontSize: 11, lineHeight: 1.6, marginBottom: 12 }}>
            Generate clustered random vectors — handy to populate a fresh scope. Each point gets a{" "}
            <span className="code">cluster</span> field for color-coding.
          </p>
          <div className="row">
            <div className="field">
              <label>count</label>
              <input
                className="input num"
                type="number"
                min={1}
                max={2000}
                value={count}
                onChange={(e) => setCount(Number(e.target.value) | 0)}
              />
            </div>
            <div className="field">
              <label>clusters</label>
              <input
                className="input num"
                type="number"
                min={1}
                max={12}
                value={clusters}
                onChange={(e) => setClusters(Number(e.target.value) | 0)}
              />
            </div>
          </div>
          <div className="field">
            <label>payload template · json</label>
            <textarea
              className="textarea"
              value={payload}
              onChange={(e) => setPayload(e.target.value)}
              style={{ fontSize: 11, minHeight: 60 }}
            />
          </div>
          <button className="btn primary" style={{ width: "100%" }} disabled={busy} onClick={submitBatch}>
            {busy ? "inserting…" : `◢ insert ${count} points`}
          </button>
        </>
      )}

      <hr className="hr" />
      <p className="muted" style={{ fontSize: 10.5, lineHeight: 1.7 }}>
        Inserts are durable (WAL-first) and land in the working set. They appear live in the scope
        immediately; <span className="code">◈ commit</span> to snapshot them as a version.
      </p>
    </div>
  );
}
