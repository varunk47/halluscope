import { fmt, type ResultSummary } from "../../api";
import { Panel, SectionLabel, Stat } from "../Primitives";

interface Reading {
  auroc: number;
  auroc_lo: number;
  auroc_hi: number;
  by_pair?: Record<string, { auroc: number; auroc_lo: number; auroc_hi: number }>;
}

/** Same probe recipe on a second model: at the matched depth, and at that model's own best layer. */
export default function TransferView({ summary }: { summary: ResultSummary }) {
  const t = summary as unknown as {
    src_model: string;
    dst_model: string;
    src_layer: number;
    src_n_layers: number;
    dst_n_layers: number;
    dst_matched_layer: number;
    dst_best_layer: number;
    cka_matched: number;
    cka_best: number;
    src_at_best: Reading;
    dst_at_matched: Reading;
    dst_at_own_best: Reading;
    dataset?: string;
  };
  const rows: [string, string, Reading, string][] = [
    ["source at its best layer", `L${t.src_layer} of ${t.src_n_layers - 1}`, t.src_at_best, ""],
    ["target at matched depth", `L${t.dst_matched_layer} of ${t.dst_n_layers - 1}`, t.dst_at_matched, fmt.num(t.cka_matched)],
    ["target at its own best", `L${t.dst_best_layer} of ${t.dst_n_layers - 1}`, t.dst_at_own_best, fmt.num(t.cka_best)],
  ];
  return (
    <div className="flex flex-col gap-4">
      <Panel>
        <div className="flex flex-wrap gap-x-10 gap-y-3">
          <Stat label="source" value={<span className="text-[14px]">{t.src_model}</span>} />
          <Stat label="target" value={<span className="text-[14px]">{t.dst_model}</span>} />
          <Stat label="dataset" value={t.dataset?.replace(/^items_?/, "") || "items"} />
          <Stat label="CKA at best layers" value={fmt.num(t.cka_best)} hint="linear, on the test items" />
        </div>
      </Panel>
      <Panel padded={false}>
        <div className="px-5 pt-4">
          <SectionLabel right="weights are not shared; the recipe is refit on each model">does the phenomenon transfer?</SectionLabel>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-[12.5px]">
            <thead>
              <tr className="label border-b border-line">
                <th className="text-left font-medium px-5 py-2">reading</th>
                <th className="text-right font-medium px-3 py-2">layer</th>
                <th className="text-right font-medium px-3 py-2">AUROC [CI]</th>
                <th className="text-right font-medium px-3 py-2">single turn</th>
                <th className="text-right font-medium px-3 py-2">multi turn</th>
                <th className="text-right font-medium px-3 py-2 pr-5">CKA</th>
              </tr>
            </thead>
            <tbody className="mono">
              {rows.map(([name, layer, m, cka]) => (
                <tr key={name} className="border-b border-line/60 last:border-b-0 text-text">
                  <td className="px-5 py-2 font-sans">{name}</td>
                  <td className="text-right px-3 py-2">{layer}</td>
                  <td className="text-right px-3 py-2">{fmt.ci(m)}</td>
                  <td className="text-right px-3 py-2">{fmt.num(m.by_pair?.single_turn_ab?.auroc)}</td>
                  <td className="text-right px-3 py-2">{fmt.num(m.by_pair?.multi_turn_cd?.auroc)}</td>
                  <td className="text-right px-3 py-2 pr-5 text-muted">{cka || "-"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
    </div>
  );
}
