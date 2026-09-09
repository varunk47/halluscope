import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { ResultSummary } from "../../api";
import { Panel, Stat } from "../Primitives";
import { AXIS, C, ChartFrame, GRID, makeTooltip } from "../charts/theme";
import { EmptyChart } from "../charts/TurnTrajectory";

const LABELS = ["specified", "underspecified", "inconsistent"];

interface BehaviorRow {
  n?: number;
  asked?: number;
  flagged?: number;
  silently_assumed?: number;
}

function isRow(v: unknown): v is BehaviorRow {
  return typeof v === "object" && v !== null && ("asked" in v || "silently_assumed" in v || "flagged" in v);
}

const Tip = makeTooltip((_n, v) => `${(Number(v) * 100).toFixed(1)}%`);

export default function BehaviorView({ summary }: { summary: ResultSummary }) {
  const s = summary.summary ?? {};
  const rows = Object.entries(s)
    .filter((e): e is [string, BehaviorRow] => isRow(e[1]))
    .sort((a, b) => {
      const ia = LABELS.indexOf(a[0]);
      const ib = LABELS.indexOf(b[0]);
      return (ia === -1 ? 99 : ia) - (ib === -1 ? 99 : ib);
    })
    .map(([label, r]) => ({
      label,
      n: r.n,
      asked: r.asked ?? 0,
      flagged: r.flagged ?? 0,
      silently_assumed: r.silently_assumed ?? 0,
    }));

  const agreement = s.judge_agreement as { n?: number; kappa_silently_assumed?: number } | undefined;
  const under = rows.find((r) => r.label === "underspecified");

  return (
    <div className="flex flex-col gap-4">
      <Panel>
        <div className="flex flex-wrap gap-x-10 gap-y-3">
          <Stat label="model" value={<span className="text-[14px]">{summary.model ?? "n/a"}</span>} />
          <Stat label="condition" value={summary.condition ?? "n/a"} />
          <Stat label="split" value={summary.split ?? "n/a"} />
          {under && (
            <Stat
              label="silently assumed, underspecified"
              value={`${(under.silently_assumed * 100).toFixed(1)}%`}
              tone="risk"
              hint={under.n !== undefined ? `n = ${under.n}` : undefined}
            />
          )}
          {agreement?.kappa_silently_assumed !== undefined && (
            <Stat
              label="judge agreement, kappa"
              value={agreement.kappa_silently_assumed.toFixed(3)}
              tone="incons"
              hint={agreement.n !== undefined ? `${agreement.n} double-judged` : "silently assumed"}
            />
          )}
        </div>
      </Panel>

      <Panel>
        <ChartFrame title="behaviour by label" right="fraction of items" height={240}>
          {rows.length === 0 ? (
            <EmptyChart text="no per-label summary in this result" />
          ) : (
            <ResponsiveContainer>
              <BarChart data={rows} margin={{ top: 8, right: 12, left: -14, bottom: 0 }} barCategoryGap="28%" barGap={3}>
                <CartesianGrid {...GRID} />
                <XAxis dataKey="label" {...AXIS} tick={{ ...AXIS.tick, fontFamily: "Inter, sans-serif", fill: C.text }} />
                <YAxis domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} {...AXIS} />
                <Tooltip content={<Tip />} cursor={{ fill: "rgba(255,255,255,0.04)" }} />
                <Legend
                  iconType="square"
                  iconSize={8}
                  wrapperStyle={{ fontSize: 11, color: C.muted, paddingTop: 6 }}
                  formatter={(v: string) => v.replace(/_/g, " ")}
                />
                <Bar dataKey="asked" fill={C.safe} fillOpacity={0.85} radius={[2, 2, 0, 0]} isAnimationActive={false} />
                <Bar dataKey="flagged" fill={C.incons} fillOpacity={0.85} radius={[2, 2, 0, 0]} isAnimationActive={false} />
                <Bar dataKey="silently_assumed" fill={C.riskHot} fillOpacity={0.85} radius={[2, 2, 0, 0]} isAnimationActive={false} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </ChartFrame>
      </Panel>

      {rows.length > 0 && (
        <Panel padded={false}>
          <div className="overflow-x-auto">
            <table className="w-full text-[12.5px]">
              <thead>
                <tr className="label border-b border-line">
                  <th className="text-left font-medium px-5 py-2">label</th>
                  <th className="text-right font-medium px-3 py-2">n</th>
                  <th className="text-right font-medium px-3 py-2">asked</th>
                  <th className="text-right font-medium px-3 py-2">flagged</th>
                  <th className="text-right font-medium px-3 py-2 pr-5">silently assumed</th>
                </tr>
              </thead>
              <tbody className="mono">
                {rows.map((r) => (
                  <tr key={r.label} className="border-b border-line/60 last:border-b-0">
                    <td className="px-5 py-2 font-sans text-text">{r.label}</td>
                    <td className="text-right px-3 py-2 text-muted">{r.n ?? "-"}</td>
                    <td className="text-right px-3 py-2 text-safe">{(r.asked * 100).toFixed(1)}%</td>
                    <td className="text-right px-3 py-2 text-incons">{(r.flagged * 100).toFixed(1)}%</td>
                    <td className="text-right px-3 py-2 pr-5 text-risk-hot">{(r.silently_assumed * 100).toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
      )}
    </div>
  );
}
