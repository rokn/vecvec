import { useState } from "react";
import { useStore } from "../store";
import { fmtNum, payloadKeys } from "../lib/format";
import { Modal } from "./Modal";

export function TopBar() {
  const stats = useStore((s) => s.stats);
  const points = useStore((s) => s.points);
  const total = useStore((s) => s.total);
  const truncated = useStore((s) => s.truncated);
  const viewVersion = useStore((s) => s.viewVersion);
  const head = useStore((s) => s.head);
  const projection = useStore((s) => s.projection);
  const setProjection = useStore((s) => s.setProjection);
  const colorKey = useStore((s) => s.colorKey);
  const setColorKey = useStore((s) => s.setColorKey);

  const [committing, setCommitting] = useState(false);

  if (!stats) return <div className="topbar" />;

  const keys = payloadKeys(points);
  const live = viewVersion == null;
  const shownVersion = viewVersion ?? head;

  return (
    <header className="topbar">
      <div className="hud-title">
        <span className="name">{stats.name}</span>
        <span className="badge metric">{stats.metric}</span>
        <span className="badge">{stats.dim}d</span>
        {shownVersion != null ? (
          <span className={`badge ${live ? "live" : "past"}`}>
            {live ? `live · v${shownVersion}` : `viewing v${shownVersion}`}
          </span>
        ) : (
          <span className="badge">uncommitted</span>
        )}
      </div>

      <div className="hud-stats">
        <div className="stat">
          <span className="k">points</span>
          <span className="v accent num">{fmtNum(stats.count)}</span>
        </div>
        <div className="stat">
          <span className="k">in scope</span>
          <span className="v num">
            {fmtNum(points.length)}
            {truncated && <span className="muted" style={{ fontSize: 10 }}> /{fmtNum(total)}</span>}
          </span>
        </div>

        <div className="stat" style={{ gap: 4 }}>
          <span className="k">projection</span>
          <div className="seg" style={{ marginTop: 2 }}>
            <button className={projection === "pca" ? "on" : ""} onClick={() => setProjection("pca")}>
              pca
            </button>
            <button
              className={projection === "umap" ? "on" : ""}
              onClick={() => setProjection("umap")}
            >
              umap
            </button>
          </div>
        </div>

        <div className="stat" style={{ gap: 4 }}>
          <span className="k">color by</span>
          <select
            className="input"
            style={{ padding: "4px 6px", fontSize: 11, marginTop: 2, minWidth: 110 }}
            value={colorKey ?? ""}
            onChange={(e) => setColorKey(e.target.value || null)}
          >
            <option value="">— none —</option>
            {keys.map((k) => (
              <option key={k} value={k}>
                {k}
              </option>
            ))}
          </select>
        </div>

        <div className="stat">
          <button className="btn primary" onClick={() => setCommitting(true)} title="snapshot working state">
            ◈ commit
          </button>
        </div>
      </div>

      {committing && <CommitModal onClose={() => setCommitting(false)} />}
    </header>
  );
}

function CommitModal({ onClose }: { onClose: () => void }) {
  const commit = useStore((s) => s.commit);
  const [message, setMessage] = useState("");
  const [tag, setTag] = useState("");
  const [busy, setBusy] = useState(false);

  const submit = async () => {
    setBusy(true);
    await commit(message.trim() || undefined, tag.trim() || undefined);
    setBusy(false);
    onClose();
  };

  return (
    <Modal title={<><b>◈</b> commit snapshot</>} onClose={onClose}>
      <p className="muted" style={{ fontSize: 11, lineHeight: 1.6, marginBottom: 14 }}>
        Seals the working set into an immutable version. Time-travel and diffs read from
        these. Deletes after a commit never change what the commit sees.
      </p>
      <div className="field">
        <label>message</label>
        <input
          className="input"
          autoFocus
          placeholder="batch 2 — refreshed embeddings"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && submit()}
        />
      </div>
      <div className="field">
        <label>tag (optional)</label>
        <input
          className="input"
          placeholder="v1.0"
          value={tag}
          onChange={(e) => setTag(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && submit()}
        />
      </div>
      <div className="wrap-actions" style={{ marginTop: 8, justifyContent: "flex-end" }}>
        <button className="btn" onClick={onClose}>
          cancel
        </button>
        <button className="btn primary" disabled={busy} onClick={submit}>
          {busy ? "committing…" : "commit"}
        </button>
      </div>
    </Modal>
  );
}
