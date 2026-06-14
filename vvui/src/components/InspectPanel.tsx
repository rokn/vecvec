import { useStore } from "../store";
import { fmtVal } from "../lib/format";

export function InspectPanel() {
  const selected = useStore((s) => s.selectedPoint);
  const points = useStore((s) => s.points);
  const deletePoint = useStore((s) => s.deletePoint);
  const queryByPoint = useStore((s) => s.queryByPoint);

  if (selected == null) {
    return (
      <div className="empty-note">
        no point selected.
        <br />
        click a node in the scope, or pick one from <span className="code">browse</span>.
      </div>
    );
  }

  const p = points.find((pt) => pt.id === selected);
  if (!p) {
    return (
      <div className="empty-note">
        point <span className="accent">#{selected}</span> is not in the current view.
        <br />
        it may be deleted or outside this version.
      </div>
    );
  }

  const maxAbs = Math.max(1e-6, ...p.vector.map((x) => Math.abs(x)));
  const entries = p.payload && typeof p.payload === "object" ? Object.entries(p.payload) : [];

  return (
    <div>
      <div className="spread" style={{ marginBottom: 14 }}>
        <span className="inspect-id">#{p.id}</span>
        <span className="badge">{p.vector.length}d</span>
      </div>

      <div className="wrap-actions" style={{ marginBottom: 16 }}>
        <button className="btn" style={{ flex: 1 }} onClick={() => queryByPoint(p.id, 12)}>
          ◎ neighbors
        </button>
        <button className="btn danger" onClick={() => deletePoint(p.id)}>
          ✕ delete
        </button>
      </div>

      <div className="subhead">payload</div>
      {entries.length === 0 ? (
        <p className="muted" style={{ fontSize: 11, marginBottom: 4 }}>
          no payload on this point.
        </p>
      ) : (
        <div className="kv" style={{ marginBottom: 6 }}>
          {entries.map(([k, v]) => (
            <span key={k} className="chip">
              <b>{k}</b>
              {fmtVal(v)}
            </span>
          ))}
        </div>
      )}

      <hr className="hr" />
      <div className="subhead">vector · {p.vector.length}</div>
      <div className="vec-grid">
        {p.vector.slice(0, 256).map((x, i) => {
          const t = Math.abs(x) / maxAbs;
          return (
            <div className="vec-cell" key={i}>
              <span className="vi num">{i}</span>
              <span className="vv num">{x.toFixed(3)}</span>
              <span
                className={`vbar ${x >= 0 ? "pos" : "neg"}`}
                style={{ width: `${Math.max(4, t * 100)}%` }}
              />
            </div>
          );
        })}
      </div>
      {p.vector.length > 256 && (
        <p className="muted" style={{ fontSize: 10, marginTop: 8 }}>
          +{p.vector.length - 256} more dims hidden.
        </p>
      )}
    </div>
  );
}
