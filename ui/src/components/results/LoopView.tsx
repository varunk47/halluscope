import type { ResultSummary } from "../../api";
import { Empty, Panel, SectionLabel, Stat } from "../Primitives";

const LABEL_ORDER = ["all", "specified", "underspecified", "inconsistent"];

interface LoopRow {
  n?: number;
  correct?: number;
  partial_or_better?: number;
  assumption_rate?: number;
  questions_per_task?: number;
}

interface FlatRow extends LoopRow {
  condition: string;
  label: string;
}

function isLoopRow(v: unknown): v is LoopRow {
  return typeof v === "object" && v !== null && ("correct" in v || "assumption_rate" in v || "questions_per_task" in v);
}

/**
 * Loop summaries come in one of two shapes: one file per condition with
 * `summary[label]`, or a merged file with `summary[condition][label]`. Both
 * flatten to (condition, label) rows.
 */
function flatten(summary: Record<string, unknown>, fallbackCondition: string): FlatRow[] {
  const rows: FlatRow[] = [];
  for (const [k, v] of Object.entries(summary)) {
    if (isLoopRow(v)) {
      rows.push({ condition: fallbackCondition, label: k, ...v });
    } else if (typeof v === "object" && v !== null) {
      for (const [label, inner] of Object.entries(v as Record<string, unknown>)) {
        if (isLoopRow(inner)) rows.push({ condition: k, label, ...inner });
      }
    }
  }
  return rows.sort((a, b) => {
    if (a.condition !== b.condition) return a.condition.localeCompare(b.condition);
    const ia = LABEL_ORDER.indexOf(a.label);
    const ib = LABEL_ORDER.indexOf(b.label);
    return (ia === -1 ? 99 : ia) - (ib === -1 ? 99 : ib);
  });
}

const pct = (v: number | undefined) => (v === undefined ? "-" : `${(v * 100).toFixed(1)}%`);

export default function LoopView({ summary }: { summary: ResultSummary }) {
  const rows = flatten(summary.summary ?? {}, summary.condition ?? "default");
  const conditions = [...new Set(rows.map((r) => r.condition))];
  const all = rows.find((r) => r.label === "all") ?? rows[0];

  return (
    <div className="flex flex-col gap-4">
      <Panel>
        <div className="flex flex-wrap gap-x-10 gap-y-3">
          <Stat label="model" value={<span className="text-[14px]">{summary.model ?? "n/a"}</span>} />
          <Stat label="condition" value={summary.condition ?? conditions.join(", ") ?? "n/a"} />
          <Stat label="split" value={summary.split ?? "n/a"} />
          {summary.seed !== undefined && summary.seed !== null && <Stat label="seed" value={summary.seed} />}
          {all && (
            <>
              <Stat label="correct" value={pct(all.correct)} tone="safe" hint={all.n !== undefined ? `n = ${all.n}` : undefined} />
              <Stat label="assumption rate" value={pct(all.assumption_rate)} tone="risk" />
              <Stat label="questions per task" value={all.questions_per_task?.toFixed(2) ?? "-"} tone="incons" />
            </>
          )}
        </div>
      </Panel>

      {rows.length === 0 ? (
        <Empty title="no loop summary in this result" />
      ) : (
        <Panel padded={false}>
          <div className="px-5 pt-4">
            <SectionLabel right={`${conditions.length} condition${conditions.length === 1 ? "" : "s"}`}>
              outcome by condition and label
            </SectionLabel>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-[12.5px]">
              <thead>
                <tr className="label border-b border-line">
                  <th className="text-left font-medium px-5 py-2">condition</th>
                  <th className="text-left font-medium px-3 py-2">label</th>
                  <th className="text-right font-medium px-3 py-2">n</th>
                  <th className="text-right font-medium px-3 py-2">correct</th>
                  <th className="text-right font-medium px-3 py-2">partial or better</th>
                  <th className="text-right font-medium px-3 py-2">assumption rate</th>
                  <th className="text-right font-medium px-3 py-2 pr-5">questions per task</th>
                </tr>
              </thead>
              <tbody className="mono">
                {rows.map((r, i) => {
                  const firstOfCondition = i === 0 || rows[i - 1].condition !== r.condition;
                  return (
                    <tr
                      key={`${r.condition}/${r.label}`}
                      className={`border-b border-line/60 last:border-b-0 ${firstOfCondition && i > 0 ? "border-t border-t-line-2" : ""}`}
                    >
                      <td className="px-5 py-2 font-sans text-text">{firstOfCondition ? r.condition : ""}</td>
                      <td className={`px-3 py-2 font-sans ${r.label === "all" ? "text-text" : "text-muted"}`}>{r.label}</td>
                      <td className="text-right px-3 py-2 text-muted">{r.n ?? "-"}</td>
                      <td className="text-right px-3 py-2 text-safe">{pct(r.correct)}</td>
                      <td className="text-right px-3 py-2">{pct(r.partial_or_better)}</td>
                      <td className="text-right px-3 py-2 text-risk">{pct(r.assumption_rate)}</td>
                      <td className="text-right px-3 py-2 pr-5 text-incons">{r.questions_per_task?.toFixed(2) ?? "-"}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </Panel>
      )}
    </div>
  );
}
