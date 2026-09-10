import type { ReactNode } from "react";
import type { TooltipProps } from "recharts";

export const C = {
  ink: "#0b0f14",
  panel: "#121820",
  line: "#1f2933",
  line2: "#2a3642",
  text: "#e6edf3",
  muted: "#8b98a5",
  dim: "#5c6873",
  safe: "#22d3ee",
  risk: "#f97316",
  riskHot: "#ef4444",
  incons: "#a78bfa",
};

export const AXIS = {
  stroke: C.line2,
  tick: { fill: C.muted, fontSize: 11, fontFamily: "JetBrains Mono, monospace" },
  tickLine: false as const,
  axisLine: { stroke: C.line2 },
};

export const GRID = { stroke: C.line, strokeDasharray: "2 4", vertical: false };

/** Colour for a probability on the risk ramp: cyan when safe, orange to red when risky. */
export function riskColor(p: number): string {
  if (p < 0.5) return C.safe;
  const t = Math.min(1, (p - 0.5) / 0.5);
  // interpolate #f97316 -> #ef4444
  const a = [249, 115, 22];
  const b = [239, 68, 68];
  const mix = a.map((x, i) => Math.round(x + (b[i] - x) * t));
  return `rgb(${mix[0]}, ${mix[1]}, ${mix[2]})`;
}

type RowFmt = (name: string, value: unknown, payload?: Record<string, unknown>) => ReactNode;

export function makeTooltip(format?: RowFmt, title?: (label: unknown) => ReactNode) {
  return function ChartTooltip(props: TooltipProps<number, string>) {
    const { active, payload, label } = props;
    if (!active || !payload || payload.length === 0) return null;
    return (
      <div className="rounded-lg border border-line-2 bg-ink/95 px-3 py-2 text-[12px] shadow-panel">
        <div className="mono text-muted mb-1">{title ? title(label) : String(label)}</div>
        {payload.map((p, i) => (
          <div key={i} className="flex items-center gap-2">
            <span className="h-1.5 w-1.5 rounded-sm" style={{ background: p.color ?? C.muted }} />
            <span className="text-muted">{p.name}</span>
            <span className="mono ml-auto text-text">
              {format
                ? format(String(p.name), p.value, p.payload as Record<string, unknown>)
                : typeof p.value === "number"
                  ? p.value.toFixed(3)
                  : String(p.value)}
            </span>
          </div>
        ))}
      </div>
    );
  };
}

export function ChartFrame({
  title,
  right,
  height = 200,
  children,
}: {
  title: string;
  right?: ReactNode;
  height?: number;
  children: ReactNode;
}) {
  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <span className="label">{title}</span>
        {right && <span className="text-[11px] text-muted mono">{right}</span>}
      </div>
      <div style={{ height }} className="w-full">
        {children}
      </div>
    </div>
  );
}
