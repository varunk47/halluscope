import { useEffect, useState } from "react";
import { NavLink } from "react-router-dom";
import { getHealth, type Health } from "../api";

const LINKS = [
  { to: "/", label: "Live" },
  { to: "/review", label: "Review" },
  { to: "/results", label: "Results" },
  { to: "/provenance", label: "Provenance" },
];

export default function TopBar() {
  const [health, setHealth] = useState<Health | null>(null);
  const [offline, setOffline] = useState(false);

  useEffect(() => {
    let alive = true;
    const tick = async () => {
      try {
        const h = await getHealth();
        if (alive) {
          setHealth(h);
          setOffline(false);
        }
      } catch {
        if (alive) setOffline(true);
      }
    };
    void tick();
    const id = window.setInterval(tick, 15000);
    return () => {
      alive = false;
      window.clearInterval(id);
    };
  }, []);

  return (
    <header className="sticky top-0 z-40 border-b border-line bg-ink/85 backdrop-blur">
      <div className="max-w-[1440px] mx-auto px-5 md:px-8 h-14 flex items-center gap-6">
        <div className="flex items-baseline gap-3 min-w-0">
          <span className="text-[17px] font-semibold tracking-tight">
            Hallu<span className="text-safe">Scope</span>
          </span>
          <span className="hidden lg:inline text-[12px] text-muted truncate">
            internal-state probes for underspecified requests
          </span>
        </div>

        <nav className="flex items-center gap-1 ml-2" aria-label="Main navigation">
          {LINKS.map((l) => (
            <NavLink
              key={l.to}
              to={l.to}
              end={l.to === "/"}
              className={({ isActive }) =>
                `relative px-3 py-1.5 text-[13px] rounded-md transition ${
                  isActive ? "text-text bg-panel-2" : "text-muted hover:text-text hover:bg-panel"
                }`
              }
            >
              {({ isActive }) => (
                <>
                  {l.label}
                  {isActive && (
                    <span className="absolute left-3 right-3 -bottom-[13px] h-px bg-safe" />
                  )}
                </>
              )}
            </NavLink>
          ))}
        </nav>

        <div className="ml-auto">
          <StatusPill health={health} offline={offline} />
        </div>
      </div>
    </header>
  );
}

function StatusPill({ health, offline }: { health: Health | null; offline: boolean }) {
  if (offline) {
    return (
      <div className="chip border-risk-hot/50 text-risk-hot bg-risk-hot/10">
        <span className="h-1.5 w-1.5 rounded-full bg-risk-hot animate-pulseDot" />
        backend offline
      </div>
    );
  }
  if (!health) {
    return (
      <div className="chip border-line text-muted">
        <span className="h-1.5 w-1.5 rounded-full bg-dim animate-pulseDot" />
        connecting
      </div>
    );
  }
  const model = health.model_loaded ?? "no model loaded";
  const fitted = health.gate?.fitted;
  return (
    <div className="flex items-center gap-2">
      <div
        className={`chip ${
          health.model_loaded
            ? "border-safe/40 text-safe bg-safe/10"
            : "border-line text-muted bg-panel"
        }`}
        title={health.model_loaded ? `model ${health.model_loaded}` : "the scorer loads the model on first request"}
      >
        <span
          className={`h-1.5 w-1.5 rounded-full ${health.model_loaded ? "bg-safe" : "bg-dim"}`}
        />
        <span className="mono max-w-[220px] truncate">{model}</span>
      </div>
      <div
        className={`chip ${
          fitted ? "border-safe/40 text-safe bg-safe/10" : "border-risk/40 text-risk bg-risk/10"
        }`}
        title={
          fitted
            ? `gate fitted at layer ${health.gate.layer}, threshold ${Number(health.gate.threshold).toFixed(3)}`
            : health.gate?.reason ?? "gate not fitted"
        }
      >
        {fitted ? "gate fitted" : "gate not fitted"}
        {fitted && health.gate.layer !== undefined && (
          <span className="mono text-[10px] opacity-80">L{health.gate.layer}</span>
        )}
      </div>
      <span className="hidden xl:inline mono text-[11px] text-dim">v{health.version}</span>
    </div>
  );
}
