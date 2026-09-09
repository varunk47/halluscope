import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { AXIS, C, ChartFrame, GRID, makeTooltip } from "./theme";
import { EmptyChart } from "./TurnTrajectory";

const Tip = makeTooltip(
  (_n, v) => `${Number(v).toFixed(3)} nats`,
  (l) => `layer ${String(l)}`,
);

export default function EntropyLine({ values }: { values: number[] }) {
  const data = values.map((v, i) => ({ layer: i, entropy: v }));
  return (
    <ChartFrame title="logit-lens entropy" right={values.length ? `${values.length} layers` : undefined} height={150}>
      {data.length === 0 ? (
        <EmptyChart text="uq disabled or no entropy returned" />
      ) : (
        <ResponsiveContainer>
          <AreaChart data={data} margin={{ top: 8, right: 10, left: -18, bottom: 0 }}>
            <defs>
              <linearGradient id="entropyFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={C.incons} stopOpacity={0.35} />
                <stop offset="100%" stopColor={C.incons} stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid {...GRID} />
            <XAxis dataKey="layer" {...AXIS} interval="preserveStartEnd" />
            <YAxis {...AXIS} width={44} tickFormatter={(v: number) => v.toFixed(1)} />
            <Tooltip content={<Tip />} />
            <Area
              type="monotone"
              dataKey="entropy"
              name="entropy"
              stroke={C.incons}
              strokeWidth={1.5}
              fill="url(#entropyFill)"
              dot={false}
              isAnimationActive={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      )}
    </ChartFrame>
  );
}
