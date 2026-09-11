import type { ResultSummary } from "../../api";
import { Panel, SectionLabel, Stat } from "../Primitives";
import { C } from "../charts/theme";

/**
 * Asking rate against push size along the gap direction. A causal test: if
 * the direction made the model ask, the right side of every row would climb.
 */
export default function SteerView({ summary }: { summary: ResultSummary }) {
  const s = summary as unknown as {
    model: string;
    layer: number;
    n_items: number;
    n_specified: number;
    n_gap: number;
    alphas: number[];
    asking_rate: Record<string, Record<string, number>>;
    dataset?: string;
  };
  const alphas = s.alphas.map((a) => `${a >= 0 ? "+" : ""}${a.toFixed(1)}`);
  const labels = Object.keys(s.asking_rate);
  return (
    <div className="flex flex-col gap-4">
      <Panel>
        <div className="flex flex-wrap gap-x-10 gap-y-3">
          <Stat label="model" value={<span className="text-[14px]">{s.model}</span>} />
          <Stat label="steered at layer" value={s.layer} />
          <Stat label="items" value={s.n_items} hint={`${s.n_specified} specified, ${s.n_gap} with a gap`} />
          <Stat label="dataset" value={s.dataset?.replace(/^items_?/, "") || "items"} />
        </div>
      </Panel>
      <Panel>
        <SectionLabel right="alpha in train-set standard deviations along the mass-mean direction">
          fraction of steered replies that ask a clarifying question
        </SectionLabel>
        <div className="overflow-x-auto">
          <table className="w-full text-[12.5px]">
            <thead>
              <tr className="label border-b border-line">
                <th className="text-left font-medium py-2">label</th>
                {alphas.map((a) => (
                  <th key={a} className="text-right font-medium px-3 py-2 mono">
                    {a}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="mono">
              {labels.map((label) => (
                <tr key={label} className="border-b border-line/60 last:border-b-0">
                  <td className="py-2 font-sans text-text">{label}</td>
                  {alphas.map((a) => {
                    const v = s.asking_rate[label]?.[a];
                    return (
                      <td key={a} className="text-right px-3 py-2">
                        <span
                          className="inline-block min-w-[44px] rounded-md px-1.5 py-0.5"
                          style={{
                            background: v === undefined ? "transparent" : `rgba(249,115,22,${Math.min(0.6, v * 1.2)})`,
                            color: v === undefined ? C.dim : C.text,
                          }}
                        >
                          {v === undefined ? "n/a" : v.toFixed(2)}
                        </span>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-3 text-[12.5px] text-muted leading-5 max-w-[70ch]">
          Pushing toward the underspecified side (positive alpha) did not raise the asking rate. The direction reads
          the state; adding it back does not make the model ask. The probe is a detector, and the gate is what turns
          detection into behavior.
        </p>
      </Panel>
    </div>
  );
}
