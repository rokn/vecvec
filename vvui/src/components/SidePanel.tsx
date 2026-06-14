import { useEffect, useState } from "react";
import { useStore } from "../store";
import { SearchPanel } from "./SearchPanel";
import { InsertPanel } from "./InsertPanel";
import { BrowsePanel } from "./BrowsePanel";
import { InspectPanel } from "./InspectPanel";

type Tab = "search" | "insert" | "browse" | "inspect";

export function SidePanel() {
  const [tab, setTab] = useState<Tab>("search");
  const selected = useStore((s) => s.selectedPoint);
  const active = useStore((s) => s.active);

  // Jump to inspect when a point gets selected (e.g. clicked in the scope).
  useEffect(() => {
    if (selected != null) setTab("inspect");
  }, [selected]);

  const tabs: { id: Tab; label: string }[] = [
    { id: "search", label: "search" },
    { id: "insert", label: "insert" },
    { id: "browse", label: "browse" },
    { id: "inspect", label: "inspect" },
  ];

  return (
    <div className="side-panel">
      <div className="side-tabs">
        {tabs.map((t) => (
          <button key={t.id} className={tab === t.id ? "on" : ""} onClick={() => setTab(t.id)}>
            {t.label}
            {t.id === "inspect" && selected != null && (
              <span style={{ color: "var(--phosphor)" }}> ●</span>
            )}
          </button>
        ))}
      </div>
      <div className="side-body">
        {!active ? (
          <div className="empty-note">no collection selected.</div>
        ) : tab === "search" ? (
          <SearchPanel />
        ) : tab === "insert" ? (
          <InsertPanel />
        ) : tab === "browse" ? (
          <BrowsePanel onInspect={() => setTab("inspect")} />
        ) : (
          <InspectPanel />
        )}
      </div>
    </div>
  );
}
