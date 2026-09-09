import { useEffect, useRef, type ReactNode } from "react";
import type { GateEvent, Item, Role, Turn } from "../api";
import { RoleTag } from "./Primitives";

export interface TurnMeta {
  mode: "ask" | "answer";
  gate: GateEvent | null;
}

/** A dialogue turn plus optional UI-only annotation from the chat gate. */
export type DialogTurn = Turn & { meta?: TurnMeta };

export function stripMeta(turns: DialogTurn[]): Turn[] {
  return turns.map(({ role, content }) => ({ role, content }));
}

interface Props {
  turns: DialogTurn[];
  onChange: (next: DialogTurn[]) => void;
  presets: Item[];
  presetId: string;
  onPreset: (id: string) => void;
  disabled?: boolean;
  trailing?: ReactNode;
}

export default function DialogueBuilder({
  turns,
  onChange,
  presets,
  presetId,
  onPreset,
  disabled,
  trailing,
}: Props) {
  const update = (i: number, patch: Partial<DialogTurn>) =>
    onChange(turns.map((t, j) => (j === i ? { ...t, ...patch } : t)));
  const remove = (i: number) => onChange(turns.filter((_, j) => j !== i));
  const add = (role: Role) => onChange([...turns, { role, content: "" }]);

  const nextRole: Role = turns.length === 0 || turns[turns.length - 1].role === "assistant" ? "user" : "assistant";

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-2">
        <select
          className="select min-w-[260px] flex-1"
          value={presetId}
          onChange={(e) => onPreset(e.target.value)}
          disabled={disabled}
          aria-label="load a preset dialogue"
        >
          <option value="">load a preset from the seed set</option>
          {presets.map((p) => (
            <option key={p.id} value={p.id}>
              {p.id} {"·"} {p.label} {"·"} {snippet(p)}
            </option>
          ))}
        </select>
        <button type="button" className="btn-ghost" onClick={() => onChange([{ role: "user", content: "" }])} disabled={disabled}>
          clear
        </button>
      </div>

      <div className="flex flex-col gap-2">
        {turns.map((t, i) => (
          <TurnEditor
            key={i}
            index={i}
            turn={t}
            isLast={i === turns.length - 1}
            disabled={disabled}
            onRole={(role) => update(i, { role })}
            onText={(content) => update(i, { content })}
            onRemove={() => remove(i)}
          />
        ))}
        {trailing}
      </div>

      <div className="flex items-center gap-2">
        <button type="button" className="btn-ghost btn-xs" onClick={() => add(nextRole)} disabled={disabled}>
          + {nextRole} turn
        </button>
        <button
          type="button"
          className="btn-ghost btn-xs"
          onClick={() => add(nextRole === "user" ? "assistant" : "user")}
          disabled={disabled}
        >
          + {nextRole === "user" ? "assistant" : "user"} turn
        </button>
        <span className="ml-auto text-[11px] text-dim">dialogue must end on a user turn</span>
      </div>
    </div>
  );
}

function snippet(item: Item): string {
  const last = item.turns[item.turns.length - 1]?.content ?? "";
  const s = last.replace(/\s+/g, " ").trim();
  return s.length > 60 ? `${s.slice(0, 60)}...` : s;
}

function TurnEditor({
  index,
  turn,
  isLast,
  disabled,
  onRole,
  onText,
  onRemove,
}: {
  index: number;
  turn: DialogTurn;
  isLast: boolean;
  disabled?: boolean;
  onRole: (r: Role) => void;
  onText: (s: string) => void;
  onRemove: () => void;
}) {
  const ref = useRef<HTMLTextAreaElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "0px";
    el.style.height = `${Math.max(56, el.scrollHeight)}px`;
  }, [turn.content]);

  const user = turn.role === "user";
  return (
    <div
      className={`group relative rounded-lg border transition ${
        user ? "border-line bg-panel" : "border-incons/20 bg-[#141626]"
      } ${isLast && user ? "focus-within:border-safe/50 focus-within:shadow-glow" : "focus-within:border-line-2"}`}
    >
      <div className="flex items-center gap-2 px-3 pt-2">
        <span className="mono text-[10px] text-dim">{String(index + 1).padStart(2, "0")}</span>
        <button
          type="button"
          className="hover:opacity-80 transition"
          title="toggle role"
          disabled={disabled}
          onClick={() => onRole(user ? "assistant" : "user")}
        >
          <RoleTag role={turn.role} />
        </button>
        {turn.meta && <MetaBadge meta={turn.meta} />}
        <button
          type="button"
          onClick={onRemove}
          disabled={disabled}
          className="ml-auto text-[11px] text-dim hover:text-risk-hot opacity-0 group-hover:opacity-100 focus:opacity-100 transition"
        >
          remove
        </button>
      </div>
      <textarea
        ref={ref}
        className="w-full resize-none bg-transparent px-3 pb-3 pt-1.5 text-[13.5px] leading-6 text-text placeholder:text-dim outline-none"
        placeholder={user ? "what does the user ask?" : "what did the assistant say?"}
        value={turn.content}
        disabled={disabled}
        onChange={(e) => onText(e.target.value)}
        rows={2}
      />
    </div>
  );
}

export function MetaBadge({ meta }: { meta: TurnMeta }) {
  const ask = meta.mode === "ask";
  return (
    <span className="flex items-center gap-1.5">
      <span className={`chip ${ask ? "border-risk/40 text-risk bg-risk/10" : "border-safe/40 text-safe bg-safe/10"}`}>
        {ask ? "clarifying question" : "answer"}
      </span>
      {meta.gate && (
        <span className="mono text-[10px] text-dim">
          gate {meta.gate.fired ? "fired" : "quiet"}
          {meta.gate.prob !== null && ` p=${meta.gate.prob.toFixed(2)}`}
        </span>
      )}
    </span>
  );
}
