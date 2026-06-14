import { useEffect, useMemo, useRef, useState } from "react";
import { useStore } from "../store";
import { project } from "../lib/projection";
import { PALETTE, fmtVal, payloadSummary, payloadValue } from "../lib/format";
import type { Payload, PointRecord } from "../types";

interface Lay {
  id: number;
  nx: number; // normalized projection coords in [0,1]
  ny: number;
  payload: Payload;
}

interface ColorModel {
  colorOf: (id: number) => string;
  legend: { label: string; color: string }[];
}

function buildColors(points: PointRecord[], key: string | null): ColorModel {
  if (!key) return { colorOf: () => PALETTE[0], legend: [] };

  const vals = new Map<number, unknown>();
  let allNumeric = true;
  let min = Infinity;
  let max = -Infinity;
  const distinct = new Set<string>();
  for (const p of points) {
    const v = payloadValue(p.payload, key);
    vals.set(p.id, v);
    if (typeof v === "number") {
      if (v < min) min = v;
      if (v > max) max = v;
    } else if (v !== undefined) {
      allNumeric = false;
    }
    distinct.add(fmtVal(v));
  }

  if (allNumeric && distinct.size > 8 && max > min) {
    const colorOf = (id: number) => {
      const v = vals.get(id);
      if (typeof v !== "number") return "#33414f";
      return lerpColor("#1b6b5c", "#7dffd9", (v - min) / (max - min));
    };
    return {
      colorOf,
      legend: [
        { label: fmtVal(min), color: "#1b6b5c" },
        { label: fmtVal(max), color: "#7dffd9" },
      ],
    };
  }

  const order = Array.from(distinct).sort();
  const idx = new Map(order.map((v, i) => [v, i] as const));
  const colorOf = (id: number) => PALETTE[(idx.get(fmtVal(vals.get(id))) ?? 0) % PALETTE.length];
  const legend = order
    .slice(0, 12)
    .map((label) => ({ label, color: PALETTE[(idx.get(label) ?? 0) % PALETTE.length] }));
  return { colorOf, legend };
}

function lerpColor(a: string, b: string, t: number): string {
  const ca = hex(a);
  const cb = hex(b);
  return `rgb(${Math.round(ca[0] + (cb[0] - ca[0]) * t)},${Math.round(
    ca[1] + (cb[1] - ca[1]) * t,
  )},${Math.round(ca[2] + (cb[2] - ca[2]) * t)})`;
}
function hex(h: string): [number, number, number] {
  const n = parseInt(h.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

export function GraphScope() {
  const points = useStore((s) => s.points);
  const projection = useStore((s) => s.projection);
  const colorKey = useStore((s) => s.colorKey);
  const highlight = useStore((s) => s.highlight);
  const diff = useStore((s) => s.diff);
  const selected = useStore((s) => s.selectedPoint);
  const selectPoint = useStore((s) => s.selectPoint);
  const queryByPoint = useStore((s) => s.queryByPoint);
  const clearHighlight = useStore((s) => s.clearHighlight);
  const loadingPoints = useStore((s) => s.loadingPoints);
  const active = useStore((s) => s.active);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const view = useRef({ zoom: 1, panX: 0, panY: 0 });
  const drag = useRef<{ x: number; y: number; moved: boolean } | null>(null);
  const sizeRef = useRef({ w: 0, h: 0, S: 0 });
  const rafRef = useRef(0);

  const [projected, setProjected] = useState<Lay[]>([]);
  const [projecting, setProjecting] = useState(false);
  const [hover, setHover] = useState<{ id: number; sx: number; sy: number } | null>(null);
  const [, setTick] = useState(0);

  const colors = useMemo(() => buildColors(points, colorKey), [points, colorKey]);

  // ── compute the 2D projection (deferred so UMAP doesn't block paint) ───────
  useEffect(() => {
    if (points.length === 0) {
      setProjected([]);
      return;
    }
    let cancelled = false;
    setProjecting(true);
    const handle = setTimeout(() => {
      const coords = project(
        projection,
        points.map((p) => p.vector),
      );
      if (cancelled) return;
      setProjected(
        points.map((p, i) => ({ id: p.id, nx: coords[i][0], ny: coords[i][1], payload: p.payload })),
      );
      setProjecting(false);
      view.current = { zoom: 1, panX: 0, panY: 0 };
    }, 16);
    return () => {
      cancelled = true;
      clearTimeout(handle);
    };
  }, [points, projection]);

  // ── canvas sizing ─────────────────────────────────────────────────────────
  useEffect(() => {
    const wrap = wrapRef.current;
    if (!wrap) return;
    const ro = new ResizeObserver(() => {
      const r = wrap.getBoundingClientRect();
      sizeRef.current = { w: r.width, h: r.height, S: Math.min(r.width, r.height) * 0.86 };
      setTick((t) => t + 1);
    });
    ro.observe(wrap);
    return () => ro.disconnect();
  }, []);

  const requestDraw = () => {
    cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(draw);
  };
  useEffect(requestDraw);

  function worldToScreen(wx: number, wy: number): [number, number] {
    const { w, h } = sizeRef.current;
    const { zoom, panX, panY } = view.current;
    return [w / 2 + zoom * (wx + panX), h / 2 + zoom * (wy + panY)];
  }
  function toScreen(nx: number, ny: number): [number, number] {
    const S = sizeRef.current.S;
    return worldToScreen((nx - 0.5) * S, (ny - 0.5) * S);
  }

  function draw() {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const { w, h, S } = sizeRef.current;
    if (w === 0 || h === 0) return;
    const dpr = window.devicePixelRatio || 1;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    const ctx = canvas.getContext("2d")!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const { zoom } = view.current;

    // world grid (parallax)
    const half = S * 0.62;
    const step = S / 8;
    ctx.lineWidth = 1;
    ctx.strokeStyle = "rgba(74,242,200,0.05)";
    ctx.beginPath();
    for (let g = -half; g <= half + 1; g += step) {
      const [sx0, sy0] = worldToScreen(g, -half);
      const [sx1, sy1] = worldToScreen(g, half);
      ctx.moveTo(sx0, sy0);
      ctx.lineTo(sx1, sy1);
      const [hx0, hy0] = worldToScreen(-half, g);
      const [hx1, hy1] = worldToScreen(half, g);
      ctx.moveTo(hx0, hy0);
      ctx.lineTo(hx1, hy1);
    }
    ctx.stroke();

    // center crosshair
    ctx.strokeStyle = "rgba(74,242,200,0.12)";
    ctx.beginPath();
    const [ccx, ccy] = worldToScreen(0, 0);
    ctx.moveTo(ccx, ccy - 9);
    ctx.lineTo(ccx, ccy + 9);
    ctx.moveTo(ccx - 9, ccy);
    ctx.lineTo(ccx + 9, ccy);
    ctx.stroke();

    const dimMode = !!highlight;
    const hl = highlight?.ids;
    const added = diff?.added;
    const byId = new Map(projected.map((p) => [p.id, p]));

    // query origin → neighbor hairlines
    if (highlight) {
      const origin = highlight.origin != null ? byId.get(highlight.origin) : null;
      if (origin) {
        const [ox, oy] = toScreen(origin.nx, origin.ny);
        ctx.strokeStyle = "rgba(74,242,200,0.22)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (const r of highlight.results) {
          const t = byId.get(r.id);
          if (!t) continue;
          const [tx, ty] = toScreen(t.nx, t.ny);
          ctx.moveTo(ox, oy);
          ctx.lineTo(tx, ty);
        }
        ctx.stroke();
      }
    }

    const r0 = Math.max(1.6, Math.min(3.6, 2.4 * Math.sqrt(zoom)));

    for (const p of projected) {
      const [sx, sy] = toScreen(p.nx, p.ny);
      if (sx < -20 || sy < -20 || sx > w + 20 || sy > h + 20) continue;
      const isHl = hl?.has(p.id);
      const isAdded = added?.has(p.id);
      const base = colors.colorOf(p.id);

      if (dimMode && !isHl && p.id !== highlight?.origin && p.id !== selected) {
        ctx.globalAlpha = 0.16;
        ctx.fillStyle = base;
        ctx.beginPath();
        ctx.arc(sx, sy, r0, 0, Math.PI * 2);
        ctx.fill();
        ctx.globalAlpha = 1;
        continue;
      }

      ctx.globalAlpha = 0.18;
      ctx.fillStyle = base;
      ctx.beginPath();
      ctx.arc(sx, sy, r0 * 2.4, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalAlpha = 1;
      ctx.beginPath();
      ctx.arc(sx, sy, r0, 0, Math.PI * 2);
      ctx.fill();

      if (isAdded && !dimMode) {
        ctx.globalAlpha = 0.9;
        ctx.strokeStyle = "#9be23c";
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.arc(sx, sy, r0 + 3, 0, Math.PI * 2);
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
    }

    if (highlight) {
      ctx.save();
      ctx.shadowColor = "rgba(74,242,200,0.9)";
      ctx.shadowBlur = 12;
      ctx.fillStyle = "#7dffd9";
      for (const r of highlight.results) {
        const t = byId.get(r.id);
        if (!t) continue;
        const [sx, sy] = toScreen(t.nx, t.ny);
        ctx.beginPath();
        ctx.arc(sx, sy, r0 + 1.2, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();

      const origin = highlight.origin != null ? byId.get(highlight.origin) : null;
      if (origin) {
        const [sx, sy] = toScreen(origin.nx, origin.ny);
        ctx.save();
        ctx.shadowColor = "rgba(255,177,98,0.9)";
        ctx.shadowBlur = 14;
        ctx.fillStyle = "#ffb162";
        ctx.beginPath();
        ctx.arc(sx, sy, r0 + 2, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      }
    }

    const sel = selected != null ? byId.get(selected) : null;
    if (sel) {
      const [sx, sy] = toScreen(sel.nx, sel.ny);
      ctx.strokeStyle = "rgba(74,242,200,0.35)";
      ctx.setLineDash([3, 4]);
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(sx, 0);
      ctx.lineTo(sx, h);
      ctx.moveTo(0, sy);
      ctx.lineTo(w, sy);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.strokeStyle = "#4af2c8";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(sx, sy, r0 + 5, 0, Math.PI * 2);
      ctx.stroke();
    }

    if (hover) {
      const hp = byId.get(hover.id);
      if (hp) {
        const [sx, sy] = toScreen(hp.nx, hp.ny);
        ctx.strokeStyle = "rgba(205,216,228,0.7)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(sx, sy, r0 + 4, 0, Math.PI * 2);
        ctx.stroke();
      }
    }
  }

  // ── interaction ───────────────────────────────────────────────────────────
  function nearestPoint(mx: number, my: number): Lay | null {
    let best: Lay | null = null;
    let bestD = 12 * 12;
    for (const p of projected) {
      const [sx, sy] = toScreen(p.nx, p.ny);
      const d = (sx - mx) * (sx - mx) + (sy - my) * (sy - my);
      if (d < bestD) {
        bestD = d;
        best = p;
      }
    }
    return best;
  }

  const onWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const rect = canvasRef.current!.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const { w, h } = sizeRef.current;
    const v = view.current;
    const newZoom = Math.max(0.4, Math.min(16, v.zoom * Math.exp(-e.deltaY * 0.0014)));
    const worldX = (mx - w / 2) / v.zoom - v.panX;
    const worldY = (my - h / 2) / v.zoom - v.panY;
    v.panX = (mx - w / 2) / newZoom - worldX;
    v.panY = (my - h / 2) / newZoom - worldY;
    v.zoom = newZoom;
    requestDraw();
  };

  const onDown = (e: React.PointerEvent) => {
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    drag.current = { x: e.clientX, y: e.clientY, moved: false };
  };
  const onMove = (e: React.PointerEvent) => {
    const rect = canvasRef.current!.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    if (drag.current) {
      const dx = e.clientX - drag.current.x;
      const dy = e.clientY - drag.current.y;
      if (Math.abs(dx) + Math.abs(dy) > 3) drag.current.moved = true;
      view.current.panX += dx / view.current.zoom;
      view.current.panY += dy / view.current.zoom;
      drag.current.x = e.clientX;
      drag.current.y = e.clientY;
      requestDraw();
      return;
    }
    const near = nearestPoint(mx, my);
    if (near) setHover({ id: near.id, sx: mx, sy: my });
    else if (hover) setHover(null);
  };
  const onUp = (e: React.PointerEvent) => {
    const wasDrag = drag.current?.moved;
    drag.current = null;
    if (wasDrag) return;
    const rect = canvasRef.current!.getBoundingClientRect();
    const near = nearestPoint(e.clientX - rect.left, e.clientY - rect.top);
    if (near) selectPoint(near.id === selected ? null : near.id);
  };
  const onDouble = (e: React.MouseEvent) => {
    const rect = canvasRef.current!.getBoundingClientRect();
    const near = nearestPoint(e.clientX - rect.left, e.clientY - rect.top);
    if (near) queryByPoint(near.id, 12);
  };

  const resetView = () => {
    view.current = { zoom: 1, panX: 0, panY: 0 };
    requestDraw();
  };

  const hoverPoint = hover ? points.find((p) => p.id === hover.id) : null;

  return (
    <div className="scope" ref={wrapRef}>
      <canvas
        ref={canvasRef}
        onWheel={onWheel}
        onPointerDown={onDown}
        onPointerMove={onMove}
        onPointerUp={onUp}
        onPointerLeave={() => setHover(null)}
        onDoubleClick={onDouble}
        style={{ cursor: drag.current ? "grabbing" : "crosshair", touchAction: "none" }}
      />

      <div className="scope-overlay">
        <div className="scope-hud-row">
          <div className="scope-readout">
            <div>
              <b>proj</b> {projection.toUpperCase()} → 2d
            </div>
            <div>
              <b>nodes</b> {projected.length}
            </div>
            {colorKey && (
              <div>
                <b>hue</b> {colorKey}
              </div>
            )}
            {highlight && (
              <div style={{ color: "var(--phosphor)" }}>
                <b style={{ color: "var(--phosphor-dim)" }}>sel</b> {highlight.label}
              </div>
            )}
          </div>
          <div className="scope-tools">
            {highlight && (
              <button className="btn sm" onClick={clearHighlight}>
                clear
              </button>
            )}
            <button className="btn sm" onClick={resetView}>
              ⤢ reset
            </button>
          </div>
        </div>

        {colors.legend.length > 0 && (
          <div className="scope-hud-row" style={{ alignItems: "flex-end" }}>
            <div
              className="kv"
              style={{
                background: "rgba(6,10,15,0.7)",
                padding: "7px 9px",
                borderRadius: 2,
                border: "1px solid var(--line)",
              }}
            >
              {colors.legend.map((l) => (
                <span
                  key={l.label}
                  className="chip"
                  style={{ display: "inline-flex", alignItems: "center", gap: 6 }}
                >
                  <span
                    style={{
                      width: 8,
                      height: 8,
                      borderRadius: 8,
                      background: l.color,
                      boxShadow: `0 0 6px ${l.color}`,
                    }}
                  />
                  {l.label}
                </span>
              ))}
            </div>
            <div className="scope-readout" style={{ textAlign: "right" }}>
              drag&nbsp;pan · scroll&nbsp;zoom · click&nbsp;select · dbl-click&nbsp;neighbors
            </div>
          </div>
        )}
      </div>

      {hover && hoverPoint && (
        <div
          className="tooltip"
          style={{
            left: Math.min(hover.sx + 14, sizeRef.current.w - 220),
            top: Math.max(hover.sy - 10, 8),
          }}
        >
          <div className="tid">#{hover.id}</div>
          <div className="tpay">{payloadSummary(hoverPoint.payload)}</div>
        </div>
      )}

      {projecting && (
        <div className="scope-empty" style={{ background: "rgba(5,8,12,0.4)" }}>
          <div className="scanbar" style={{ width: 180 }} />
          <div className="big">projecting · {projection}</div>
        </div>
      )}

      {!projecting && projected.length === 0 && (
        <div className="scope-empty">
          <div className="big">
            {!active ? "no collection" : loadingPoints ? "loading…" : "empty scope"}
          </div>
          <div className="muted" style={{ maxWidth: 280, lineHeight: 1.7 }}>
            {!active
              ? "select or create a collection to begin."
              : "no points in this view. insert vectors from the panel on the right."}
          </div>
        </div>
      )}
    </div>
  );
}
