import { useMemo, useState } from "react";
import { useStore } from "../store";
import { clockTime, relTime } from "../lib/format";
import type { VersionInfo } from "../types";
import { Modal } from "./Modal";

export function Timeline() {
  const versions = useStore((s) => s.versions);
  const head = useStore((s) => s.head);
  const viewVersion = useStore((s) => s.viewVersion);
  const setViewVersion = useStore((s) => s.setViewVersion);
  const diff = useStore((s) => s.diff);
  const stats = useStore((s) => s.stats);
  const tagVersion = useStore((s) => s.tagVersion);
  const branchVersion = useStore((s) => s.branchVersion);
  const restoreVersion = useStore((s) => s.restoreVersion);

  const [refModal, setRefModal] = useState<null | "tag" | "branch">(null);
  const [confirmRestore, setConfirmRestore] = useState(false);

  const sorted = useMemo(
    () => [...versions].sort((a, b) => a.version - b.version),
    [versions],
  );

  const focused = viewVersion ?? head;
  const focusedInfo = sorted.find((v) => v.version === focused) ?? null;

  const positions = useMemo(() => layoutTicks(sorted), [sorted]);

  if (!stats) return <div className="timeline" />;

  const live = viewVersion == null;
  const idx = focused != null ? sorted.findIndex((v) => v.version === focused) : -1;

  return (
    <div className="timeline">
      <div className="timeline-head">
        <span className="panel-title">
          <b>◷</b> timeline
          <span className="muted" style={{ marginLeft: 8, fontWeight: 400, letterSpacing: 0 }}>
            {sorted.length} version{sorted.length === 1 ? "" : "s"}
          </span>
        </span>

        <div className="diff-readout">
          {diff && (
            <>
              <span className="add">
                <span className="glyph">+</span>
                {diff.added.size}
              </span>
              <span className="del">
                <span className="glyph">−</span>
                {diff.removed.length}
              </span>
              <span className="muted" style={{ fontSize: 10 }}>
                Δ vs parent
              </span>
            </>
          )}
        </div>

        <div className="wrap-actions">
          <span className={`badge ${live ? "live" : "past"}`}>
            {live ? "● live" : `◀ v${focused}`}
          </span>
          {focusedInfo && (
            <>
              <button className="btn sm" onClick={() => setRefModal("tag")}>
                tag
              </button>
              <button className="btn sm" onClick={() => setRefModal("branch")}>
                branch
              </button>
              <button className="btn sm" onClick={() => setConfirmRestore(true)} disabled={live && focused === head}>
                restore
              </button>
            </>
          )}
          {!live && (
            <button className="btn sm primary" onClick={() => setViewVersion(null)}>
              → live
            </button>
          )}
        </div>
      </div>

      {sorted.length === 0 ? (
        <div className="empty-note" style={{ padding: "10px 0" }}>
          no commits yet — <span className="code">◈ commit</span> to start the timeline. then scrub
          to time-travel.
        </div>
      ) : (
        <div className="timeline-track-wrap">
          <div className="timeline-axis">
            <div className="timeline-baseline" />
            {sorted.map((v, i) => {
              const isFocus = v.version === focused;
              const isHead = v.version === head;
              const stem = isFocus ? 42 : 28;
              return (
                <div
                  key={v.version}
                  className={`tick-node ${isFocus ? "sel" : ""} ${isHead ? "head" : ""}`}
                  style={{ left: `${positions[i]}%` }}
                  onClick={() => setViewVersion(isHead ? null : v.version)}
                  title={v.message ?? `version ${v.version}`}
                >
                  {(v.trigger === "manual" || isHead) && (
                    <span className="tick-tag" style={{ color: isHead ? "var(--add)" : undefined }}>
                      {isHead ? "head" : ""}
                    </span>
                  )}
                  <span className="tick-stem" style={{ height: stem }} />
                  <span className="tick-dot" />
                  <span className="tick-label">v{v.version}</span>
                </div>
              );
            })}
          </div>

          <div className="spread" style={{ marginTop: 22 }}>
            <input
              className="scrubber"
              type="range"
              min={0}
              max={Math.max(0, sorted.length - 1)}
              value={idx < 0 ? sorted.length - 1 : idx}
              onChange={(e) => {
                const v = sorted[Number(e.target.value)];
                if (v) setViewVersion(v.version === head ? null : v.version);
              }}
            />
            <span
              className="muted num"
              style={{ marginLeft: 14, fontSize: 10.5, minWidth: 150, textAlign: "right" }}
            >
              {focusedInfo
                ? `${clockTime(focusedInfo.created_at_ms)} · ${relTime(focusedInfo.created_at_ms)}`
                : "—"}
            </span>
          </div>

          {focusedInfo?.message && (
            <div className="muted" style={{ fontSize: 11, marginTop: 6, fontStyle: "italic" }}>
              “{focusedInfo.message}” · <span className="dim">{focusedInfo.trigger}</span>
            </div>
          )}
        </div>
      )}

      {refModal && focused != null && (
        <RefModal
          kind={refModal}
          version={focused}
          onClose={() => setRefModal(null)}
          onSubmit={(nm) => (refModal === "tag" ? tagVersion(nm, focused) : branchVersion(nm, focused))}
        />
      )}

      {confirmRestore && focused != null && (
        <Modal title="restore version" onClose={() => setConfirmRestore(false)} width={400}>
          <p className="dim" style={{ lineHeight: 1.7 }}>
            Restore the working set to <span className="accent">v{focused}</span>? This is a forward
            commit re-pointing live state at that snapshot — older versions stay queryable.
          </p>
          <div className="wrap-actions" style={{ marginTop: 18, justifyContent: "flex-end" }}>
            <button className="btn" onClick={() => setConfirmRestore(false)}>
              cancel
            </button>
            <button
              className="btn primary"
              onClick={() => {
                restoreVersion(focused);
                setConfirmRestore(false);
              }}
            >
              restore v{focused}
            </button>
          </div>
        </Modal>
      )}
    </div>
  );
}

function RefModal({
  kind,
  version,
  onClose,
  onSubmit,
}: {
  kind: "tag" | "branch";
  version: number;
  onClose: () => void;
  onSubmit: (name: string) => void;
}) {
  const [name, setName] = useState("");
  return (
    <Modal title={`${kind} → v${version}`} onClose={onClose} width={360}>
      <div className="field">
        <label>{kind} name</label>
        <input
          className="input"
          autoFocus
          placeholder={kind === "tag" ? "v1.0" : "experiment"}
          value={name}
          onChange={(e) => setName(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && name.trim()) {
              onSubmit(name.trim());
              onClose();
            }
          }}
        />
      </div>
      <div className="wrap-actions" style={{ justifyContent: "flex-end" }}>
        <button className="btn" onClick={onClose}>
          cancel
        </button>
        <button
          className="btn primary"
          disabled={!name.trim()}
          onClick={() => {
            onSubmit(name.trim());
            onClose();
          }}
        >
          set {kind}
        </button>
      </div>
    </Modal>
  );
}

/** Lay out version ticks by wall-clock time, clamped so labels stay readable. */
function layoutTicks(versions: VersionInfo[]): number[] {
  if (versions.length === 0) return [];
  if (versions.length === 1) return [50];
  const times = versions.map((v) => v.created_at_ms);
  const min = Math.min(...times);
  const max = Math.max(...times);
  const span = max - min;
  return versions.map((v, i) => {
    const t = span > 0 ? (v.created_at_ms - min) / span : i / (versions.length - 1);
    return 4 + t * 92;
  });
}
