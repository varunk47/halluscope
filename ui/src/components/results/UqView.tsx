import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ErrorBar,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { fmt, type ResultSummary } from "../../api";
import { Panel, SectionLabel, Stat } from "../Primitives";
import { AXIS, C, ChartFrame, GRID, makeTooltip } from "../charts/theme";
import { EmptyChart } from "../charts/TurnTrajectory";

interface UqMethod {
  auroc?: number;
  auroc_lo?: number;
  auroc_hi?: number;
  n_generations?: number;
  mean_seconds?: number;
  n?: number;
  [k: string]: unknown;
}

function isMethod(v: unknown): v is UqMethod {
  return typeof v === "object" && v !== null && typeof (v as UqMethod).auroc === "number";
}

const Tip = makeTooltip(
  (_n, v, p) => `${Number(v).toFixed(3)} [${Number(p?.lo).toFixed(2)}, ${Number(p?.hi).toFixed(2)}]`,
  (l) => String(l).replace(/_/g, " "),
);

export default function UqView({ summary }: { summary: ResultSummary }) {
  const methods = Object.entries(summary.summary ?? {})
    .filter((e): e is [string, UqMethod] => isMethod(e[1]))
    .map(([name, m]) => ({
      name,
      auroc: m.auroc as number,
      lo: m.auroc_lo ?? m.auroc ?? 0,
      hi: m.auroc_hi ?? m.auroc ?? 0,
      err: [
        Math.max(0, (m.auroc ?? 0) - (m.auroc_lo ?? m.auroc ?? 0)),
        Math.max(0, (m.auroc_hi ?? m.auroc ?? 0) - (m.auroc ?? 0)),
      ],
      n_generations: m.n_generations,
      mean_seconds: m.mean_seconds,
      n: m.n,
    }))
    .sort((a, b) => b.auroc - a.auroc);

  const height = Math.max(160, 34 * methods.length + 40);

  return (
    <div className="flex flex-col gap-4">
      <Panel>
        <div className="flex flex-wrap gap-x-10 gap-y-3">
          <Stat label="model" value={<span className="text-[14px]">{summary.model ?? "n/a"}</span>} />
          <Stat label="split" value={summary.split ?? "n/a"} />
          <Stat label="methods" value={methods.length} />
          <Stat label="best AUROC" value={methods[0] ? methods[0].auroc.toFixed(3) : "n/a"} tone="safe" hint={methods[0]?.name.replace(/_/g, " ")} />
        </div>
      </Panel>

      <Panel>
        <ChartFrame title="uncertainty signals, AUROC for underspecified" right="bootstrap 95% CI" height={height}>
          {methods.length === 0 ? (
            <EmptyChart text="no uq methods in this result" />
          ) : (
            <ResponsiveContainer>
              <BarChart data={methods} layout="vertical" margin={{ top: 4, right: 24, left: 8, bottom: 0 }} barCategoryGap="28%">
                <CartesianGrid {...GRID} vertical horizontal={false} />
                <XAxis type="number" domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} {...AXIS} />
                <YAxis
                  type="category"
                  dataKey="name"
                  width={170}
                  {...AXIS}
                  tick={{ ...AXIS.tick, fontFamily: "Inter, sans-serif", fill: C.text }}
                  tickFormatter={(v: string) => v.replace(/_/g, " ")}
                />
                <Tooltip content={<Tip />} cursor={{ fill: "rgba(255,255,255,0.04)" }} />
                <ReferenceLine x={0.5} stroke={C.dim} strokeDasharray="3 3" />
                <Bar dataKey="auroc" name="auroc" radius={[0, 2, 2, 0]} isAnimationActive={false}>
                  {methods.map((m) => (
                    <Cell key={m.name} fill={m.auroc >= 0.75 ? C.safe : m.auroc >= 0.6 ? C.incons : C.risk} fillOpacity={0.8} />
                  ))}
                  <ErrorBar dataKey="err" width={5} strokeWidth={1.2} stroke={C.text} direction="x" />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
        </ChartFrame>
      </Panel>

      <Panel padded={false}>
        <div className="px-5 pt-4">
          <SectionLabel>method table</SectionLabel>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-[12.5px]">
            <thead>
              <tr className="label border-b border-line">
                <th className="text-left font-medium px-5 py-2">method</th>
                <th className="text-right font-medium px-3 py-2">AUROC [CI]</th>
                <th className="text-right font-medium px-3 py-2">n</th>
                <th className="text-right font-medium px-3 py-2">generations</th>
                <th className="text-right font-medium px-3 py-2 pr-5">mean seconds</th>
              </tr>
            </thead>
            <tbody className="mono">
              {methods.map((m) => (
                <tr key={m.name} className="border-b border-line/60 last:border-b-0">
                  <td className="px-5 py-2 font-sans text-text">{m.name.replace(/_/g, " ")}</td>
                  <td className="text-right px-3 py-2">{fmt.ci({ auroc: m.auroc, auroc_lo: m.lo, auroc_hi: m.hi })}</td>
                  <td className="text-right px-3 py-2 text-muted">{m.n ?? "-"}</td>
                  <td className="text-right px-3 py-2">{m.n_generations ?? "-"}</td>
                  <td className="text-right px-3 py-2 pr-5">{m.mean_seconds !== undefined ? m.mean_seconds.toFixed(2) : "-"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
    </div>
  );
}
