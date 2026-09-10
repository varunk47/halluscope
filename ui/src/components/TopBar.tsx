import { useEffect, useRef, useState } from "react";
import { NavLink, useLocation } from "react-router-dom";
import { getHealth, type Health } from "../api";
import { gsap, reducedMotion } from "../lib/motion";

const LINKS = [
  { to: "/", label: "Live" },
  { to: "/review", label: "Review" },
  { to: "/results", label: "Results" },
  { to: "/provenance", label: "Provenance" },
];

export default function TopBar() {
  const [health, setHealth] = useState<Health | null>(null);
  const [offline, setOffline] = useState(false);
  const nav = useRef<HTMLElement>(null);
  const pill = useRef<HTMLSpanElement>(null);
  const { pathname } = useLocation();

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

  // One pill slides between links instead of each link drawing its own underline.
  useEffect(() => {
    const el = nav.current?.querySelector<HTMLElement>('[aria-current="page"]');
    if (!el || !pill.current || !nav.current) return;
    const x = el.offsetLeft;
    const w = el.offsetWidth;
    if (reducedMotion() || !pill.current.dataset.placed) {
      gsap.set(pill.current, { x, width: w, autoAlpha: 1 });
      pill.current.dataset.placed = "1";
      return;
    }
    gsap.to(pill.current, { x, width: w, duration: 0.35, ease: "power3.out", overwrite: true });
  }, [pathname]);

  return (
    <header
      className="sticky top-0 z-40 border-b border-line bg-ink/85 backdrop-blur"
      style={{ viewTransitionName: "topbar" }}
    >
      <div className="max-w-[1440px] mx-auto px-5 md:px-8 h-14 flex items-center gap-6">
        <NavLink to="/" className="flex items-center gap-2.5 min-w-0">
          <Mark />
          <span className="font-display text-[19px] font-semibold tracking-[-0.01em]">HalluScope</span>
        </NavLink>

        <nav ref={nav} className="relative flex items-center gap-1 ml-2" aria-label="Main navigation">
          <span
            ref={pill}
            aria-hidden="true"
            className="absolute left-0 top-0 h-full rounded-lg bg-panel-2 opacity-0 pointer-events-none"
          />
          {LINKS.map((l) => (
            <NavLink
              key={l.to}
              to={l.to}
              end={l.to === "/"}
              className={({ isActive }) =>
                `relative z-10 px-3 py-1.5 text-[13.5px] rounded-lg transition-colors ${
                  isActive ? "text-text" : "text-muted hover:text-text"
                }`
              }
            >
              {l.label}
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

/** A residual stream read at one depth: a stack of layers with one lit. */
function Mark() {
  return (
    <svg width="22" height="22" viewBox="0 0 22 22" aria-hidden="true">
      {[3, 7, 11, 15].map((y, i) => (
        <rect key={y} x="3" y={y} width="16" height="2.5" rx="1" fill={i === 2 ? "#22d3ee" : "#2a3644"} />
      ))}
      <circle cx="19.5" cy="12.25" r="2" fill="#22d3ee" />
    </svg>
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
    return <div className="chip border-line shimmer-text">connecting</div>;
  }
  const model = health.model_loaded ?? "model loads on first score";
  const fitted = health.gate?.fitted;
  return (
    <div className="flex items-center gap-2">
      <div
        className={`chip ${health.model_loaded ? "border-safe/40 text-safe bg-safe/10" : "border-line text-muted"}`}
        title={health.model_loaded ? `model ${health.model_loaded}` : "the scorer loads the model on first request"}
      >
        <span className="mono max-w-[220px] truncate">{model}</span>
      </div>
      <div
        className={`chip ${fitted ? "border-safe/40 text-safe bg-safe/10" : "border-risk/40 text-risk bg-risk/10"}`}
        title={
          fitted
            ? `gate fitted at layer ${health.gate.layer}, threshold ${Number(health.gate.threshold).toFixed(3)}`
            : (health.gate?.reason ?? "gate not fitted")
        }
      >
        {fitted ? "gate fitted" : "gate not fitted"}
        {fitted && health.gate.layer !== undefined && (
          <span className="mono text-[10px] opacity-80">L{health.gate.layer}</span>
        )}
      </div>
    </div>
  );
}
