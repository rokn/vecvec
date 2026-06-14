import { useMemo, useState } from "react";
import { useStore } from "../store";
import { payloadSummary, vecPreview } from "../lib/format";

export function BrowsePanel({ onInspect }: { onInspect: () => void }) {
  const points = useStore((s) => s.points);
  const total = useStore((s) => s.total);
  const truncated = useStore((s) => s.truncated);
  const selected = useStore((s) => s.selectedPoint);
  const selectPoint = useStore((s) => s.selectPoint);
  const deletePoint = useStore((s) => s.deletePoint);

  const [q, setQ] = useState("");

  const filtered = useMemo(() => {
    const t = q.trim().toLowerCase();
    if (!t) return points;
    return points.filter(
      (p) =>
        String(p.id).includes(t) ||
        (p.payload && JSON.stringify(p.payload).toLowerCase().includes(t)),
    );
  }, [points, q]);

  return (
    <div>
      <div className="subhead">
        browse · {points.length}
        {truncated && <span className="muted"> /{total}</span>}
      </div>

      <input
        className="input"
        placeholder="filter by id or payload…"
        value={q}
        onChange={(e) => setQ(e.target.value)}
        style={{ marginBottom: 10 }}
      />

      {truncated && (
        <p className="muted" style={{ fontSize: 10, marginBottom: 8 }}>
          showing first {points.length} of {total} points (projection cap).
        </p>
      )}

      {points.length === 0 ? (
        <div className="empty-note">no points in this view.</div>
      ) : (
        <div style={{ maxHeight: "calc(100vh - 220px)", overflow: "auto", margin: "0 -14px" }}>
          <table className="tbl">
            <thead>
              <tr>
                <th>id</th>
                <th>vector</th>
                <th>payload</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {filtered.slice(0, 400).map((p) => (
                <tr
                  key={p.id}
                  className={selected === p.id ? "sel" : ""}
                  onClick={() => {
                    selectPoint(p.id);
                    onInspect();
                  }}
                >
                  <td className="cid num">#{p.id}</td>
                  <td className="cvec num">{vecPreview(p.vector, 3)}</td>
                  <td className="cpay">{payloadSummary(p.payload)}</td>
                  <td style={{ textAlign: "right" }}>
                    <button
                      className="btn ghost sm"
                      style={{ color: "var(--del)", padding: "2px 6px" }}
                      onClick={(e) => {
                        e.stopPropagation();
                        deletePoint(p.id);
                      }}
                      title="delete point"
                    >
                      ✕
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {filtered.length > 400 && (
            <p className="muted" style={{ fontSize: 10, padding: "8px 14px" }}>
              +{filtered.length - 400} more — refine the filter to narrow.
            </p>
          )}
        </div>
      )}
    </div>
  );
}
