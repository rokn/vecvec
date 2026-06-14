import { useStore } from "../store";

export function Toasts() {
  const toasts = useStore((s) => s.toasts);
  const dismiss = useStore((s) => s.dismissToast);
  return (
    <div className="toasts">
      {toasts.map((t) => (
        <div
          key={t.id}
          className={`toast ${t.kind === "err" ? "err" : t.kind === "ok" ? "ok" : ""}`}
          onClick={() => dismiss(t.id)}
        >
          <div style={{ flex: 1 }}>
            <div className="tk">{t.title}</div>
            <div style={{ color: "var(--ink)", marginTop: 2 }}>{t.msg}</div>
          </div>
        </div>
      ))}
    </div>
  );
}
