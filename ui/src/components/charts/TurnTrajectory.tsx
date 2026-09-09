import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { AXIS, C, ChartFrame, GRID, makeTooltip } from "./theme";

const Tip = makeTooltip(
  (_n, v) => `${(Number(v) * 100).toFixed(1)}%`,
  (l) => `user turn ${String(l)}`,
);

export default function TurnTrajectory({ probs }: { probs: number[] }) {
  const data = probs.map((p, i) => ({ turn: i + 1, prob: p }));
  return (
    <ChartFrame title="trajectory across turns" right={`${probs.length} user turn${probs.length === 1 ? "" : "s"}`} height={150}>
      {data.length === 0 ? (
        <EmptyChart text="no per-turn scores" />
      ) : (
        <ResponsiveContainer>
          <LineChart data={data} margin={{ top: 8, right: 10, left: -18, bottom: 0 }}>
            <CartesianGrid {...GRID} />
            <XAxis dataKey="turn" {...AXIS} allowDecimals={false} />
            <YAxis domain={[0, 1]} ticks={[0, 0.5, 1]} {...AXIS} />
            <Tooltip content={<Tip />} />
            <ReferenceLine y={0.5} stroke={C.muted} strokeDasharray="3 3" />
            <Line
              type="monotone"
              dataKey="prob"
              name="P(underspecified)"
              stroke={C.risk}
              strokeWidth={2}
              dot={{ r: 3, fill: C.ink, stroke: C.risk, strokeWidth: 2 }}
              activeDot={{ r: 4, fill: C.risk }}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      )}
    </ChartFrame>
  );
}

export function EmptyChart({ text }: { text: string }) {
  return (
    <div className="h-full w-full grid place-items-center rounded-md border border-dashed border-line text-[12px] text-dim">
      {text}
    </div>
  );
}
