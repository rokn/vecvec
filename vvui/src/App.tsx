import { useEffect } from "react";
import { useStore } from "./store";
import { Sidebar } from "./components/Sidebar";
import { TopBar } from "./components/TopBar";
import { GraphScope } from "./components/GraphScope";
import { Timeline } from "./components/Timeline";
import { SidePanel } from "./components/SidePanel";
import { Toasts } from "./components/Toasts";

export default function App() {
  const connected = useStore((s) => s.connected);
  const init = useStore((s) => s.init);

  useEffect(() => {
    init();
  }, [init]);

  if (connected === null) {
    return (
      <div className="center-msg">
        <div className="scanbar" style={{ width: 200 }} />
        <div className="kicker">linking to vecvec · rest :6333</div>
      </div>
    );
  }

  if (connected === false) {
    return <Offline onRetry={init} />;
  }

  return (
    <>
      <div className="app">
        <Sidebar />
        <div className="main">
          <TopBar />
          <div className="workspace">
            <div className="canvas-col reveal">
              <GraphScope />
              <Timeline />
            </div>
            <div className="side-col reveal" style={{ animationDelay: "60ms" }}>
              <SidePanel />
            </div>
          </div>
        </div>
      </div>
      <Toasts />
    </>
  );
}

function Offline({ onRetry }: { onRetry: () => void }) {
  return (
    <div className="center-msg" style={{ textAlign: "center" }}>
      <div style={{ maxWidth: 460 }}>
        <h1
          className="display"
          style={{ fontSize: 30, letterSpacing: "0.04em", color: "var(--ink)", marginBottom: 6 }}
        >
          vecvec<span style={{ color: "var(--del)" }}>//</span>scope
        </h1>
        <div className="kicker" style={{ color: "var(--del)" }}>
          ● no signal · server offline
        </div>
        <p className="dim" style={{ margin: "22px 0 14px", lineHeight: 1.8 }}>
          Couldn't reach the vecvec REST gateway. Start the server, then retry:
        </p>
        <div
          className="panel"
          style={{ textAlign: "left", padding: "14px 16px", fontSize: 12, lineHeight: 1.9 }}
        >
          <div className="muted"># from the vecvec repo root</div>
          <div className="accent">cargo run -p vecvec-server</div>
          <div style={{ height: 8 }} />
          <div className="muted"># optional: load a demo dataset</div>
          <div className="accent">cd vvui && npm run seed</div>
        </div>
        <button className="btn primary" style={{ marginTop: 22 }} onClick={onRetry}>
          ↻ retry connection
        </button>
      </div>
    </div>
  );
}
