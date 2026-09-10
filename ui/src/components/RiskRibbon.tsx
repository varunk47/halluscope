import { useEffect, useRef } from "react";
import type { ScoreResponse } from "../api";
import { gsap, reducedMotion, rollNumber, useGSAP } from "../lib/motion";
import { riskColor } from "./charts/theme";

/**
 * The instrument's face. One large figure, the probability at the best
 * validation layer, rolls up from zero when a score lands; the marker slides
 * along the scale to meet it; the verdict stamps in last. The conformal gate
 * thresholds the raw decision score, not this probability, so that threshold
 * is shown beside the figure rather than on the scale.
 */
export default function RiskRibbon({ score }: { score: ScoreResponse }) {
  const scope = useRef<HTMLDivElement>(null);
  const numRef = useRef<HTMLSpanElement>(null);
  const p = score.prob_best;
  const pct = p === null ? null : Math.max(0, Math.min(1, p));
  const color = pct === null ? "#8a97a6" : riskColor(pct);
  const decisionBest = score.per_layer.find((l) => l.layer === score.best_layer)?.decision ?? null;

  useEffect(() => {
    if (pct !== null) rollNumber(numRef.current, pct * 100, (v) => v.toFixed(1));
  }, [pct]);

  useGSAP(
    () => {
      if (pct === null) return;
      if (reducedMotion()) {
        gsap.set(".marker", { left: `${pct * 100}%` });
        gsap.set(".verdict", { scale: 1, autoAlpha: 1 });
        return;
      }
      const tl = gsap.timeline();
      tl.to(".marker", { left: `${pct * 100}%`, duration: 0.9, ease: "power3.out" }, 0);
      tl.fromTo(
        ".verdict",
        { scale: 0.85, autoAlpha: 0 },
        { scale: 1, autoAlpha: 1, duration: 0.45, ease: "back.out(2)" },
        0.55,
      );
    },
    { scope, dependencies: [pct, score.gate_fired] },
  );

  return (
    <div ref={scope} className="relative overflow-hidden rounded-xl border border-line bg-panel shadow-readout">
      <div
        className="absolute inset-0 opacity-[0.14] pointer-events-none transition-colors duration-700"
        style={{ background: `radial-gradient(520px 180px at 12% 0%, ${color}, transparent 70%)` }}
      />
      <div className="relative px-5 pt-5 pb-4 flex flex-wrap items-end gap-x-8 gap-y-4">
        <div className="min-w-[210px]">
          <div className="label">
            underspecified, read at layer <span className="mono text-text">{score.best_layer ?? "?"}</span>
          </div>
          <div className="flex items-baseline gap-1.5 mt-1">
            <span
              ref={numRef}
              data-value="0"
              className="font-display mono text-[64px] leading-none font-semibold tracking-[-0.03em] transition-colors duration-700"
              style={{ color }}
            >
              {pct === null ? "n/a" : "0.0"}
            </span>
            {pct !== null && <span className="mono text-[22px] text-muted">%</span>}
          </div>
        </div>

        <div className="verdict">
          <GatePill fired={score.gate_fired} />
        </div>

        <div className="ml-auto grid grid-cols-2 gap-x-6 gap-y-2 text-right">
          <Mini label="decision" value={decisionBest === null ? "n/a" : decisionBest.toFixed(3)} />
          <Mini label="conformal threshold" value={score.threshold === null ? "n/a" : score.threshold.toFixed(3)} />
          <Mini label="model" value={score.model || "n/a"} wide />
        </div>
      </div>

      <div className="relative px-5 pb-5">
        <div className="relative h-2 rounded-full bg-ribbon-scale opacity-90">
          <div className="absolute top-[-5px] bottom-[-5px] w-px bg-text/70" style={{ left: "50%" }} />
          {pct !== null && (
            <div
              className="marker absolute -top-[5px] h-[18px] w-[3px] -ml-[1.5px] rounded-sm bg-white shadow-[0_0_0_2px_#0a0e14]"
              style={{ left: 0 }}
            />
          )}
        </div>
        <div className="mt-2 flex justify-between mono text-[10px] text-dim">
          <span>0 specified</span>
          <span className="text-muted">0.5 boundary</span>
          <span>1 underspecified</span>
        </div>
      </div>

      <div className="absolute right-4 top-3 mono text-[11px] text-dim">{score.seconds.toFixed(2)}s</div>
    </div>
  );
}

function Mini({ label, value, wide = false }: { label: string; value: string; wide?: boolean }) {
  return (
    <div className={wide ? "col-span-2" : ""}>
      <div className="text-[11px] text-dim">{label}</div>
      <div className="mono text-[13px] text-text truncate max-w-[260px]" title={value}>
        {value}
      </div>
    </div>
  );
}

export function GatePill({ fired }: { fired: boolean | null }) {
  if (fired === null) {
    return (
      <div className="chip border-line-2 text-muted !px-3 !py-1.5 !text-[12.5px]">gate not fitted</div>
    );
  }
  if (fired) {
    return (
      <div className="chip border-risk/70 text-risk bg-risk/10 !px-3 !py-1.5 !text-[12.5px]">gate fired, ask first</div>
    );
  }
  return (
    <div className="chip border-safe/50 text-safe bg-safe/10 !px-3 !py-1.5 !text-[12.5px]">gate quiet, answer</div>
  );
}
