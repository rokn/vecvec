import { useState } from "react";
import { useStore } from "../store";
import { fmtScore } from "../lib/format";
import type { Filter, FilterCondition } from "../types";

type Op = "match" | "gt" | "gte" | "lt" | "lte";
interface Cond {
  key: string;
  op: Op;
  value: string;
}

function buildFilter(conds: Cond[]): Filter | undefined {
  const must: FilterCondition[] = [];
  for (const c of conds) {
    if (!c.key.trim()) continue;
    const raw = c.value.trim();
    const num = Number(raw);
    const isNum = raw !== "" && !Number.isNaN(num);
    if (c.op === "match") {
      must.push({ key: c.key.trim(), match: isNum ? num : raw });
    } else {
      if (!isNum) continue;
      must.push({ key: c.key.trim(), range: { [c.op]: num } });
    }
  }
  return must.length ? { must } : undefined;
}

export function SearchPanel() {
  const stats = useStore((s) => s.stats);
  const selected = useStore((s) => s.selectedPoint);
  const points = useStore((s) => s.points);
  const runQuery = useStore((s) => s.runQuery);
  const runRecommend = useStore((s) => s.runRecommend);
  const queryByPoint = useStore((s) => s.queryByPoint);
  const highlight = useStore((s) => s.highlight);
  const selectPoint = useStore((s) => s.selectPoint);

  const [k, setK] = useState(12);
  const [conds, setConds] = useState<Cond[]>([]);
  const [positive, setPositive] = useState("");
  const [negative, setNegative] = useState("");

  if (!stats) return null;
  const dim = stats.dim;

  const filter = buildFilter(conds);

  const knnRandom = () => {
    const vec = Array.from({ length: dim }, () => Math.random() * 2 - 1);
    runQuery(vec, k, filter, "random vector");
  };
  const knnSelected = () => {
    if (selected != null) queryByPoint(selected, k);
  };

  const ids = (s: string) =>
    s
      .split(/[\s,]+/)
      .map((x) => x.trim())
      .filter(Boolean)
      .map(Number)
      .filter((n) => Number.isFinite(n));

  const recommend = () => {
    const pos = ids(positive);
    const neg = ids(negative);
    if (pos.length + neg.length === 0) return;
    runRecommend(pos, neg, k, filter);
  };

  return (
    <div>
      <div className="subhead">k-nn query</div>

      <div className="field">
        <label>k · results</label>
        <input
          className="input num"
          type="number"
          min={1}
          max={200}
          value={k}
          onChange={(e) => setK(Math.max(1, Number(e.target.value) | 0))}
        />
      </div>

      <div className="wrap-actions" style={{ marginBottom: 6 }}>
        <button className="btn" style={{ flex: 1 }} onClick={knnRandom}>
          ⟢ random query
        </button>
        <button className="btn" style={{ flex: 1 }} disabled={selected == null} onClick={knnSelected}>
          ◎ from #{selected ?? "—"}
        </button>
      </div>
      <p className="muted" style={{ fontSize: 10, marginBottom: 4 }}>
        tip — double-click any node in the scope to find its neighbors.
      </p>

      <hr className="hr" />
      <div className="subhead">recommend by example</div>
      <div className="field">
        <label>positive ids · move toward</label>
        <input
          className="input"
          placeholder="3, 14, 92"
          value={positive}
          onChange={(e) => setPositive(e.target.value)}
        />
      </div>
      <div className="field">
        <label>negative ids · move away</label>
        <input
          className="input"
          placeholder="7, 51"
          value={negative}
          onChange={(e) => setNegative(e.target.value)}
        />
      </div>
      <button className="btn" style={{ width: "100%" }} onClick={recommend}>
        ✦ recommend
      </button>

      <hr className="hr" />
      <div className="subhead">payload filter</div>
      {conds.length === 0 && (
        <p className="muted" style={{ fontSize: 10.5, marginBottom: 8 }}>
          no conditions — searching all points. add one to constrain by payload.
        </p>
      )}
      <div className="stack">
        {conds.map((c, i) => (
          <div className="row" key={i} style={{ gap: 6, alignItems: "center" }}>
            <input
              className="input"
              placeholder="key"
              style={{ flex: 1.2, padding: "6px 8px" }}
              value={c.key}
              onChange={(e) => setConds(upd(conds, i, { key: e.target.value }))}
            />
            <select
              className="input"
              style={{ flex: 0.9, padding: "6px 4px" }}
              value={c.op}
              onChange={(e) => setConds(upd(conds, i, { op: e.target.value as Op }))}
            >
              <option value="match">=</option>
              <option value="gt">&gt;</option>
              <option value="gte">≥</option>
              <option value="lt">&lt;</option>
              <option value="lte">≤</option>
            </select>
            <input
              className="input"
              placeholder="value"
              style={{ flex: 1, padding: "6px 8px" }}
              value={c.value}
              onChange={(e) => setConds(upd(conds, i, { value: e.target.value }))}
            />
            <button className="btn ghost sm" onClick={() => setConds(conds.filter((_, j) => j !== i))}>
              ✕
            </button>
          </div>
        ))}
      </div>
      <button
        className="btn ghost sm"
        style={{ marginTop: 8 }}
        onClick={() => setConds([...conds, { key: "", op: "match", value: "" }])}
      >
        + condition
      </button>

      {highlight && (
        <>
          <hr className="hr" />
          <div className="subhead">
            results · {highlight.results.length}
            <span className="muted" style={{ marginLeft: "auto", fontWeight: 400, letterSpacing: 0 }}>
              {highlight.label}
            </span>
          </div>
          <div className="stack" style={{ gap: 2 }}>
            {highlight.results.map((r, i) => {
              const max = highlight.results[0]?.score ?? 1;
              const min = highlight.results[highlight.results.length - 1]?.score ?? 0;
              const t = max === min ? 1 : (r.score - min) / (max - min);
              const present = points.some((p) => p.id === r.id);
              return (
                <div
                  key={r.id}
                  className={`result ${selected === r.id ? "sel" : ""}`}
                  onClick={() => selectPoint(r.id)}
                  title={present ? "" : "not in current scope view"}
                >
                  <span className="rk num">{i + 1}</span>
                  <span className="rid num">#{r.id}</span>
                  <span className="rbar">
                    <span style={{ width: `${Math.max(6, t * 100)}%` }} />
                  </span>
                  <span className="rscore num">{fmtScore(r.score)}</span>
                </div>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}

function upd(arr: Cond[], i: number, patch: Partial<Cond>): Cond[] {
  return arr.map((c, j) => (j === i ? { ...c, ...patch } : c));
}
