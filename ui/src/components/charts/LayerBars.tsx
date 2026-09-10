import { useRef, useState } from "react";
import type { LayerScore } from "../../api";
import { gsap, reducedMotion, useGSAP } from "../../lib/motion";
import { ChartFrame, riskColor } from "./theme";

/**
 * One bar per layer, embedding layer on the left, final layer on the right.
 * The bars rise out of the baseline in order, so the eye reads the sweep the
 * way the forward pass produced it. Built from plain elements rather than a
 * chart library so GSAP can drive transform alone and nothing relayouts.
 */
export default function LayerBars({
  data,
  bestLayer,
}: {
  data: LayerScore[];
  bestLayer: number | null;
}) {
  const scope = useRef<HTMLDivElement>(null);
  const [hover, setHover] = useState<LayerScore | null>(null);
  const key = data.map((d) => d.prob.toFixed(3)).join(",");

  useGSAP(
    () => {
      const bars = gsap.utils.toArray<HTMLElement>(".layer-bar");
      if (reducedMotion()) {
        gsap.set(bars, { scaleY: 1 });
        return;
      }
      gsap.fromTo(
        bars,
        { scaleY: 0 },
        { scaleY: 1, duration: 0.55, ease: "power3.out", stagger: { each: 0.018, from: "start" } },
      );
    },
    { scope, dependencies: [key] },
  );

  const shown = hover ?? data.find((d) => d.layer === bestLayer) ?? null;

  return (
    <ChartFrame
      title="P(underspecified) at every layer"
      right={
        shown ? (
          <span>
            layer {shown.layer}
            <span className="text-text ml-2">{(shown.prob * 100).toFixed(1)}%</span>
            <span className="text-dim ml-2">decision {shown.decision.toFixed(2)}</span>
          </span>
        ) : undefined
      }
      height={170}
    >
      <div ref={scope} className="relative h-full w-full" onMouseLeave={() => setHover(null)}>
        {/* the probe's 0.5 boundary */}
        <div className="absolute left-0 right-0 top-1/2 h-px bg-muted/40 border-t border-dashed border-muted/40" />
        <div className="absolute inset-0 flex items-end gap-[3px]">
          {data.map((d) => {
            const best = d.layer === bestLayer;
            return (
              <div
                key={d.layer}
                className="relative flex-1 h-full flex items-end cursor-crosshair"
                onMouseEnter={() => setHover(d)}
                title={`layer ${d.layer}: ${(d.prob * 100).toFixed(1)}%`}
              >
                <div
                  className="layer-bar w-full rounded-t-[2px]"
                  style={{
                    height: `${Math.max(2, d.prob * 100)}%`,
                    background: riskColor(d.prob),
                    opacity: best ? 1 : hover?.layer === d.layer ? 0.9 : 0.42,
                    boxShadow: best ? "0 0 0 1px #e8eef4 inset" : undefined,
                  }}
                />
                {best && (
                  <div className="absolute -top-4 left-1/2 -translate-x-1/2 mono text-[10px] text-text whitespace-nowrap">
                    L{d.layer}
                  </div>
                )}
              </div>
            );
          })}
        </div>
        <div className="absolute -bottom-5 left-0 right-0 flex justify-between mono text-[10px] text-dim">
          <span>embedding</span>
          <span>final layer</span>
        </div>
      </div>
    </ChartFrame>
  );
}
