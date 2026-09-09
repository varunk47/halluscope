import type { ScoreResponse } from "../api";
import { riskColor } from "./charts/theme";

/**
 * The headline read-out for a scored dialogue: the probability at the best
 * layer as a large number, a scale with a marker at the probe's 0.5 boundary,
 * and the gate verdict. The conformal gate itself thresholds the raw decision
 * score, so that threshold is shown alongside the probability rather than on
 * the probability scale.
 */
export default function RiskRibbon({ score }: { score: ScoreResponse }) {
  const p = score.prob_best;
  const pct = p === null ? null : Math.max(0, Math.min(1, p));
  const color = pct === null ? "#8b98a5" : riskColor(pct);
  const decisionBest = score.per_layer.find((l) => l.layer === score.best_layer)?.decision ?? null;

  return (
    <div className="relative overflow-hidden rounded-lg border border-line bg-panel shadow-panel">
      <div
        className="absolute inset-0 opacity-[0.12] pointer-events-none"
        style={{
          background: `radial-gradient(600px 160px at 20% 0%, ${color}, transparent 70%)`,
        }}
      />
      <div className="relative p-5 flex flex-wrap items-end gap-x-8 gap-y-4">
        <div className="min-w-[200px]">
          <div className="label">
            P(underspecified) at layer{" "}
            <span className="mono text-text">{score.best_layer ?? "?"}</span>
          </div>
          <div className="flex items-baseline gap-2 mt-1">
            <span
              className="mono text-[52px] leading-none font-medium tracking-tight transition-colors"
              style={{ color }}
            >
              {pct === null ? "n/a" : (pct * 100).toFixed(1)}
            </span>
            {pct !== null && <span className="mono text-[20px] text-muted">%</span>}
          </div>
        </div>

        <GatePill fired={score.gate_fired} />

        <div className="ml-auto grid grid-cols-2 gap-x-6 gap-y-2 text-right">
          <Mini label="decision" value={decisionBest === null ? "n/a" : decisionBest.toFixed(3)} />
          <Mini
            label="conformal thr"
            value={score.threshold === null ? "n/a" : score.threshold.toFixed(3)}
          />
          <Mini label="model" value={score.model || "n/a"} wide />
        </div>
      </div>

      <div className="relative px-5 pb-5">
        <div className="relative h-2 rounded-full bg-ribbon-scale opacity-90">
          {/* threshold marker at the probe's 0.5 boundary */}
          <div className="absolute top-[-5px] bottom-[-5px] w-px bg-text/70" style={{ left: "50%" }} />
          {pct !== null && (
            <div
              className="absolute -top-[5px] h-[18px] w-[3px] rounded-sm bg-white shadow-[0_0_0_2px_#0b0f14] transition-[left]"
              style={{ left: `calc(${pct * 100}% - 1.5px)` }}
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
      <div className="label !text-[10px]">{label}</div>
      <div className="mono text-[13px] text-text truncate max-w-[260px]" title={value}>
        {value}
      </div>
    </div>
  );
}

export function GatePill({ fired }: { fired: boolean | null }) {
  if (fired === null) {
    return (
      <div className="chip border-line-2 text-muted !px-3 !py-1.5 !text-[12px]">
        <span className="h-1.5 w-1.5 rounded-full bg-dim" />
        gate not fitted
      </div>
    );
  }
  if (fired) {
    return (
      <div className="chip border-risk/60 text-risk bg-risk/10 shadow-glow-risk !px-3 !py-1.5 !text-[12px] uppercase tracking-wider">
        <span className="h-1.5 w-1.5 rounded-full bg-risk animate-pulseDot" />
        gate fired: ask
      </div>
    );
  }
  return (
    <div className="chip border-safe/50 text-safe bg-safe/10 !px-3 !py-1.5 !text-[12px]">
      <span className="h-1.5 w-1.5 rounded-full bg-safe" />
      gate quiet: answer
    </div>
  );
}
