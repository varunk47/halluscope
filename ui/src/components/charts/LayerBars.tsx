import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { LayerScore } from "../../api";
import { AXIS, C, ChartFrame, GRID, makeTooltip, riskColor } from "./theme";

const Tip = makeTooltip(
  (name, value, p) =>
    name === "prob"
      ? `${(Number(value) * 100).toFixed(1)}%  (decision ${Number(p?.decision ?? 0).toFixed(2)})`
      : String(value),
  (l) => `layer ${String(l)}`,
);

export default function LayerBars({
  data,
  bestLayer,
}: {
  data: LayerScore[];
  bestLayer: number | null;
}) {
  return (
    <ChartFrame title="P(underspecified) per layer" right={bestLayer !== null ? `best L${bestLayer}` : undefined} height={190}>
      <ResponsiveContainer>
        <BarChart data={data} margin={{ top: 6, right: 6, left: -18, bottom: 0 }} barCategoryGap="18%">
          <CartesianGrid {...GRID} />
          <XAxis dataKey="layer" {...AXIS} interval="preserveStartEnd" />
          <YAxis domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} {...AXIS} />
          <Tooltip content={<Tip />} cursor={{ fill: "rgba(255,255,255,0.04)" }} />
          <ReferenceLine y={0.5} stroke={C.muted} strokeDasharray="3 3" />
          <Bar dataKey="prob" name="prob" radius={[2, 2, 0, 0]} isAnimationActive={false}>
            {data.map((d) => {
              const best = d.layer === bestLayer;
              return (
                <Cell
                  key={d.layer}
                  fill={riskColor(d.prob)}
                  fillOpacity={best ? 1 : 0.45}
                  stroke={best ? C.text : "none"}
                  strokeWidth={best ? 1 : 0}
                />
              );
            })}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </ChartFrame>
  );
}
