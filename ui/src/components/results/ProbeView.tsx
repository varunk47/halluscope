import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ErrorBar,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import { fmt, type HeadlineRow, type Metric, type ProbeFull, type ResultSummary } from "../../api";
import { useEffect, useRef } from "react";
import { rollNumber } from "../../lib/motion";
import { Panel, SectionLabel, Stat } from "../Primitives";
import { AXIS, C, ChartFrame, GRID, makeTooltip } from "../charts/theme";
import { EmptyChart } from "../charts/TurnTrajectory";

const PROBE_ORDER = ["linear", "massmean", "mlp"];
const BASELINE_ORDER = ["tfidf_dialogue", "tfidf_final_turn", "length"];

function ordered<T>(obj: Record<string, T> | undefined, order: string[]): [string, T][] {
  if (!obj) return [];
  const keys = Object.keys(obj).sort((a, b) => {
    const ia = order.indexOf(a);
    const ib = order.indexOf(b);
    return (ia === -1 ? 99 : ia) - (ib === -1 ? 99 : ib) || a.localeCompare(b);
  });
  return keys.map((k) => [k, obj[k]]);
}

export default function ProbeView({ summary, full }: { summary: ResultSummary; full: ProbeFull | null }) {
  const probes = ordered(summary.probes, PROBE_ORDER);
  const baselines = ordered(summary.baselines, BASELINE_ORDER);
  const linear = full?.probes?.linear;
  const bestTest = probes.reduce<number>((m, [, p]) => Math.max(m, p.test?.auroc ?? 0), 0);

  return (
    <div className="flex flex-col gap-4">
      {summary.headline && summary.headline.length > 0 && <Headline rows={summary.headline} />}
      <Panel>
        <div className="flex flex-wrap gap-x-10 gap-y-3">
          <Stat label="model" value={<span className="text-[14px]">{summary.model ?? "n/a"}</span>} />
          <Stat label="dataset" value={summary.dataset?.replace(/^items_?/, "") || "items"} />
          <Stat label="target" value={summary.target ?? "n/a"} />
          <Stat label="pooling" value={summary.pooling ?? "n/a"} />
          <Stat label="layers" value={summary.n_layers ?? "n/a"} />
          {summary.split_sizes && (
            <Stat
              label="split"
              value={Object.entries(summary.split_sizes)
                .map(([k, v]) => `${k[0]}${v}`)
                .join(" / ")}
              hint="train / val / test"
            />
          )}
          <Stat
            label="best test AUROC"
            value={bestTest ? bestTest.toFixed(3) : "n/a"}
            hint="separability alone; the paired delta above is the finding"
          />
        </div>
      </Panel>

      <Panel padded={false}>
        <div className="px-5 pt-4">
          <SectionLabel right="test split, bootstrap 95% CI">probes and baselines</SectionLabel>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-[12.5px]">
            <thead>
              <tr className="label border-b border-line">
                <th className="text-left font-medium px-5 py-2">probe</th>
                <th className="text-right font-medium px-3 py-2">layer</th>
                <th className="text-right font-medium px-3 py-2">AUROC [CI]</th>
                <th className="text-right font-medium px-3 py-2">AUPRC</th>
                <th className="text-right font-medium px-3 py-2">acc</th>
                <th className="text-right font-medium px-3 py-2 pr-5">ECE</th>
              </tr>
            </thead>
            <tbody className="mono">
              {probes.map(([name, p]) => (
                <MetricRow key={name} name={name} layer={p.best_layer} m={p.test} highlight={p.test?.auroc === bestTest} />
              ))}
              {baselines.map(([name, m]) => (
                <MetricRow key={name} name={name} m={m} muted />
              ))}
              {probes.length === 0 && (
                <tr>
                  <td colSpan={6} className="px-5 py-4 text-dim">
                    no probes in this result
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </Panel>

      {!full ? (
        <>
          <div className="skel h-[268px]" />
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div className="skel h-[268px]" />
            <div className="skel h-[268px]" />
          </div>
        </>
      ) : (
        <>
          <Panel>
            <LayerSweep
              val={linear?.per_layer_val_auroc}
              test={linear?.per_layer_test_auroc}
              best={linear?.best_layer ?? null}
            />
          </Panel>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Panel>
              <ReliabilityDiagram rel={linear?.test?.reliability} ece={linear?.test?.ece} />
            </Panel>
            <Panel>
              <Loto loto={full.loto ?? summary.loto} />
            </Panel>
          </div>
        </>
      )}
    </div>
  );
}

const PAIR_TITLE: Record<string, string> = {
  single_turn_ab: "single turn: specified vs underspecified",
  multi_turn_cd: "multi turn: consistent vs contradicted",
};
const PAIR_NOTE: Record<string, string> = {
  single_turn_ab: "decidable from the words by construction, so a sanity check",
  multi_turn_cd: "answerable only from the earlier turn, so the real question",
};

/**
 * The finding, before the table. A bare AUROC says how separable the labels
 * are; only the paired delta against a bag of words says whether the hidden
 * state adds anything over the text. Shown for the linear probe; the other
 * two are in the full JSON.
 */
function Headline({ rows }: { rows: HeadlineRow[] }) {
  const linear = rows.filter((r) => r.probe === "linear");
  if (linear.length === 0) return null;
  return (
    <Panel>
      <SectionLabel right={`paired bootstrap against ${linear[0].baseline.replace(/_/g, " ")}`}>
        does the hidden state beat the words?
      </SectionLabel>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {linear.map((r) => {
          const gain = r.delta > 0;
          const clear = r.p < 0.05;
          const tone = !gain ? "text-risk-hot" : clear ? "text-safe" : "text-text";
          return (
            <div key={r.pair} className="rounded-lg border border-line bg-ink/50 p-4">
              <div className="text-[13px] text-text">{PAIR_TITLE[r.pair] ?? r.pair}</div>
              <div className="text-[11.5px] text-dim mt-0.5">{PAIR_NOTE[r.pair] ?? ""}</div>
              <div className="mt-3 flex items-end gap-6">
                <div>
                  <Delta value={r.delta} className={`font-display mono text-[36px] leading-none font-semibold tracking-tight ${tone}`} />
                  <div className="mono text-[11px] text-dim mt-1">
                    [{r.lo >= 0 ? "+" : ""}
                    {r.lo.toFixed(3)}, {r.hi >= 0 ? "+" : ""}
                    {r.hi.toFixed(3)}] p={r.p.toFixed(3)} n={r.n}
                  </div>
                </div>
                <dl className="ml-auto grid grid-cols-2 gap-x-5 text-right">
                  <dt className="text-[11px] text-dim">probe</dt>
                  <dt className="text-[11px] text-dim">words</dt>
                  <dd className="mono text-[15px] text-text">{fmt.num(r.probe_auroc)}</dd>
                  <dd className="mono text-[15px] text-muted">{fmt.num(r.words_auroc)}</dd>
                </dl>
              </div>
            </div>
          );
        })}
      </div>
    </Panel>
  );
}

/** The delta rolls to its value once, so the eye lands on the finding as it settles. */
function Delta({ value, className }: { value: number; className: string }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    rollNumber(ref.current, value, (v) => `${v >= 0 ? "+" : ""}${v.toFixed(3)}`, 1.1);
  }, [value]);
  return (
    <div ref={ref} data-value="0" className={className}>
      +0.000
    </div>
  );
}

function MetricRow({
  name,
  layer,
  m,
  muted = false,
  highlight = false,
}: {
  name: string;
  layer?: number;
  m: Metric | undefined;
  muted?: boolean;
  highlight?: boolean;
}) {
  return (
    <tr className={`border-b border-line/60 last:border-b-0 ${muted ? "text-muted" : "text-text"}`}>
      <td className="px-5 py-2 font-sans">
        <span className={highlight ? "text-safe" : ""}>{name.replace(/_/g, " ")}</span>
        {muted && <span className="ml-2 text-[11px] text-dim">baseline</span>}
      </td>
      <td className="text-right px-3 py-2">{layer ?? "-"}</td>
      <td className="text-right px-3 py-2">{fmt.ci(m)}</td>
      <td className="text-right px-3 py-2">{fmt.num(m?.auprc)}</td>
      <td className="text-right px-3 py-2">{fmt.num(m?.accuracy)}</td>
      <td className="text-right px-3 py-2 pr-5">{fmt.num(m?.ece)}</td>
    </tr>
  );
}

const SweepTip = makeTooltip((_n, v) => Number(v).toFixed(3), (l) => `layer ${String(l)}`);

function LayerSweep({
  val,
  test,
  best,
}: {
  val?: Record<string, number>;
  test?: Record<string, number>;
  best: number | null;
}) {
  const layers = new Set<number>();
  Object.keys(val ?? {}).forEach((k) => layers.add(Number(k)));
  Object.keys(test ?? {}).forEach((k) => layers.add(Number(k)));
  const data = [...layers]
    .sort((a, b) => a - b)
    .map((l) => ({ layer: l, val: val?.[String(l)], test: test?.[String(l)] }));
  return (
    <ChartFrame title="layer sweep, linear probe AUROC" right={best !== null ? `best L${best}` : undefined} height={220}>
      {data.length === 0 ? (
        <EmptyChart text="no per-layer AUROC in this result" />
      ) : (
        <ResponsiveContainer>
          <LineChart data={data} margin={{ top: 8, right: 12, left: -14, bottom: 0 }}>
            <CartesianGrid {...GRID} />
            <XAxis dataKey="layer" {...AXIS} interval="preserveStartEnd" />
            <YAxis domain={[0.4, 1]} ticks={[0.5, 0.6, 0.7, 0.8, 0.9, 1]} {...AXIS} />
            <Tooltip content={<SweepTip />} />
            <ReferenceLine y={0.5} stroke={C.dim} strokeDasharray="3 3" />
            {best !== null && <ReferenceLine x={best} stroke={C.text} strokeOpacity={0.35} />}
            <Line type="monotone" dataKey="val" name="validation" stroke={C.safe} strokeWidth={2} dot={false} isAnimationActive={false} />
            <Line
              type="monotone"
              dataKey="test"
              name="test"
              stroke={C.risk}
              strokeWidth={1.5}
              strokeDasharray="5 4"
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      )}
    </ChartFrame>
  );
}

const RelTip = makeTooltip(
  (n, v, p) => (n === "accuracy" ? `${Number(v).toFixed(3)}  (n=${String(p?.count ?? "?")})` : Number(v).toFixed(3)),
  (l) => `confidence ${Number(l).toFixed(2)}`,
);

function ReliabilityDiagram({ rel, ece }: { rel?: { confidence: number[]; accuracy: number[]; count: number[] }; ece?: number }) {
  const data = rel
    ? rel.confidence.map((c, i) => ({ confidence: c, accuracy: rel.accuracy[i], count: rel.count[i] ?? 0 }))
    : [];
  const maxCount = Math.max(1, ...data.map((d) => d.count));
  return (
    <ChartFrame title="reliability, linear probe" right={ece !== undefined ? `ECE ${ece.toFixed(3)}` : undefined} height={220}>
      {data.length === 0 ? (
        <EmptyChart text="no reliability bins in this result" />
      ) : (
        <ResponsiveContainer>
          <ScatterChart margin={{ top: 8, right: 12, left: -14, bottom: 0 }}>
            <CartesianGrid {...GRID} vertical />
            <XAxis dataKey="confidence" type="number" domain={[0, 1]} ticks={[0, 0.5, 1]} {...AXIS} name="confidence" />
            <YAxis dataKey="accuracy" type="number" domain={[0, 1]} ticks={[0, 0.5, 1]} {...AXIS} name="accuracy" />
            <ZAxis dataKey="count" range={[30, 30 + 220 * Math.min(1, 8 / maxCount)]} />
            <Tooltip content={<RelTip />} cursor={{ strokeDasharray: "3 3", stroke: C.line2 }} />
            <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke={C.dim} strokeDasharray="3 3" />
            <Scatter data={data} name="accuracy" fill={C.safe} line={{ stroke: C.safe, strokeWidth: 1.5 }} isAnimationActive={false} />
          </ScatterChart>
        </ResponsiveContainer>
      )}
    </ChartFrame>
  );
}

const LotoTip = makeTooltip(
  (_n, v, p) => `${Number(v).toFixed(3)} [${Number(p?.lo).toFixed(2)}, ${Number(p?.hi).toFixed(2)}]`,
  (l) => `held out ${String(l)}`,
);

function Loto({ loto }: { loto?: Record<string, { auroc: number; auroc_lo: number; auroc_hi: number }> }) {
  const data = Object.entries(loto ?? {})
    .map(([topic, v]) => ({
      topic,
      auroc: v.auroc,
      lo: v.auroc_lo,
      hi: v.auroc_hi,
      err: [Math.max(0, v.auroc - v.auroc_lo), Math.max(0, v.auroc_hi - v.auroc)],
    }))
    .sort((a, b) => b.auroc - a.auroc);
  return (
    <ChartFrame title="leave-one-topic-out AUROC" right={data.length ? `${data.length} topics` : undefined} height={220}>
      {data.length === 0 ? (
        <EmptyChart text="no leave-one-topic-out sweep in this result" />
      ) : (
        <ResponsiveContainer>
          <BarChart data={data} margin={{ top: 8, right: 12, left: -14, bottom: 0 }} barCategoryGap="30%">
            <CartesianGrid {...GRID} />
            <XAxis dataKey="topic" {...AXIS} interval={0} tick={{ ...AXIS.tick, fontSize: 10 }} />
            <YAxis domain={[0.4, 1]} ticks={[0.5, 0.75, 1]} {...AXIS} />
            <Tooltip content={<LotoTip />} cursor={{ fill: "rgba(255,255,255,0.04)" }} />
            <ReferenceLine y={0.5} stroke={C.dim} strokeDasharray="3 3" />
            <Bar dataKey="auroc" name="auroc" radius={[2, 2, 0, 0]} isAnimationActive={false}>
              {data.map((d) => (
                <Cell key={d.topic} fill={d.auroc >= 0.75 ? C.safe : d.auroc >= 0.6 ? C.risk : C.riskHot} fillOpacity={0.8} />
              ))}
              <ErrorBar dataKey="err" width={4} strokeWidth={1.2} stroke={C.text} direction="y" />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      )}
    </ChartFrame>
  );
}
