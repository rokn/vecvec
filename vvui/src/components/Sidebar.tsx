import { useState } from "react";
import { useStore } from "../store";
import { fmtNum } from "../lib/format";
import type { Metric } from "../types";
import { Modal } from "./Modal";

export function Sidebar() {
  const collections = useStore((s) => s.collections);
  const active = useStore((s) => s.active);
  const connected = useStore((s) => s.connected);
  const select = useStore((s) => s.selectCollection);
  const drop = useStore((s) => s.dropCollection);

  const [creating, setCreating] = useState(false);
  const [confirmDrop, setConfirmDrop] = useState<string | null>(null);

  return (
    <aside className="sidebar">
      <div className="brand">
        <h1>
          vecvec<span>//</span>scope
        </h1>
        <div className="sub">
          <span className={`dot ${connected ? "" : "down"}`} />
          {connected === null ? "linking…" : connected ? "rest · :6333" : "offline"}
        </div>
      </div>

      <div className="col-list section-label" style={{ borderBottom: "1px solid var(--line)" }}>
        <span className="kicker">Collections · {collections.length}</span>
        <button className="btn sm" onClick={() => setCreating(true)}>
          + new
        </button>
      </div>

      <div className="col-list">
        {collections.length === 0 && (
          <div className="empty-note">
            no collections yet.
            <br />
            create one, or run
            <br />
            <span className="code">npm run seed</span>
          </div>
        )}
        {collections.map((c) => (
          <button
            key={c.name}
            className={`col-item ${active === c.name ? "active" : ""}`}
            onClick={() => select(c.name)}
          >
            <span className="spread">
              <span className="col-name">
                {active === c.name && <span style={{ color: "var(--phosphor)" }}>▸</span>}
                {c.name}
              </span>
              <span
                className="btn ghost sm icon"
                onClick={(e) => {
                  e.stopPropagation();
                  setConfirmDrop(c.name);
                }}
                title="drop collection"
                style={{ color: "var(--ink-faint)" }}
              >
                ✕
              </span>
            </span>
            <span className="col-meta">
              <span>
                <b>{c.dim}</b>d
              </span>
              <span>{c.metric}</span>
              <span>
                <b>{fmtNum(c.count)}</b> pts
              </span>
              <span>{c.head_version != null ? `v${c.head_version}` : "—"}</span>
            </span>
          </button>
        ))}
      </div>

      <div
        style={{
          padding: "12px 16px",
          borderTop: "1px solid var(--line)",
          fontSize: 9.5,
          letterSpacing: "0.1em",
          color: "var(--ink-faint)",
          textTransform: "uppercase",
        }}
      >
        git-like vector store · time-travel · hnsw
      </div>

      {creating && <CreateModal onClose={() => setCreating(false)} />}
      {confirmDrop && (
        <Modal title="drop collection" onClose={() => setConfirmDrop(null)} width={380}>
          <p className="dim" style={{ lineHeight: 1.7 }}>
            Permanently delete <span className="accent">{confirmDrop}</span> and all of its
            segments, versions, and WAL from disk? This cannot be undone.
          </p>
          <div className="wrap-actions" style={{ marginTop: 18, justifyContent: "flex-end" }}>
            <button className="btn" onClick={() => setConfirmDrop(null)}>
              cancel
            </button>
            <button
              className="btn danger"
              onClick={() => {
                drop(confirmDrop);
                setConfirmDrop(null);
              }}
            >
              drop it
            </button>
          </div>
        </Modal>
      )}
    </aside>
  );
}

function CreateModal({ onClose }: { onClose: () => void }) {
  const create = useStore((s) => s.createCollection);
  const [name, setName] = useState("");
  const [dim, setDim] = useState(32);
  const [metric, setMetric] = useState<Metric>("cosine");
  const [busy, setBusy] = useState(false);

  const submit = async () => {
    if (!name.trim() || dim < 1) return;
    setBusy(true);
    const ok = await create(name.trim(), dim, metric);
    setBusy(false);
    if (ok) onClose();
  };

  return (
    <Modal title={<><b>◢</b> new collection</>} onClose={onClose}>
      <div className="field">
        <label>name</label>
        <input
          className="input"
          autoFocus
          value={name}
          placeholder="embeddings"
          onChange={(e) => setName(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && submit()}
        />
      </div>
      <div className="row">
        <div className="field">
          <label>dimensions</label>
          <input
            className="input num"
            type="number"
            min={1}
            value={dim}
            onChange={(e) => setDim(Math.max(1, Number(e.target.value) | 0))}
          />
        </div>
        <div className="field">
          <label>metric</label>
          <select
            className="input"
            value={metric}
            onChange={(e) => setMetric(e.target.value as Metric)}
          >
            <option value="cosine">cosine</option>
            <option value="dot">dot</option>
            <option value="euclidean">euclidean</option>
          </select>
        </div>
      </div>
      <p className="muted" style={{ fontSize: 10.5, lineHeight: 1.6, marginTop: 2 }}>
        {metric === "cosine"
          ? "vectors are L2-normalized at ingest; similarity is a dot product."
          : metric === "dot"
            ? "raw inner product; higher is closer."
            : "squared L2 distance; lower is closer."}
      </p>
      <div className="wrap-actions" style={{ marginTop: 16, justifyContent: "flex-end" }}>
        <button className="btn" onClick={onClose}>
          cancel
        </button>
        <button className="btn primary" disabled={busy || !name.trim()} onClick={submit}>
          {busy ? "creating…" : "create"}
        </button>
      </div>
    </Modal>
  );
}
