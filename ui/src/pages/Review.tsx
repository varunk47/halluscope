import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ApiError, getItems, patchItem, type Item, type ItemPatch, type ReviewStatus, type Turn } from "../api";
import { Empty, LabelChip, PageHeader, Panel, RoleTag, Spinner, StatusChip } from "../components/Primitives";
import { useToast } from "../components/Toast";

const TOPICS = ["finetuning", "rag", "evaluation", "agents", "prompting", "serving", "data", "safety"];
const LABELS = ["specified", "underspecified", "inconsistent"];
const STATUSES = ["pending", "approved", "edited", "rejected"];
const SOURCES = ["seed", "augmented"];
const REVIEWER_KEY = "halluscope.reviewer";

interface Filters {
  status: string;
  topic: string;
  label: string;
  source: string;
  q: string;
}

const NO_FILTERS: Filters = { status: "", topic: "", label: "", source: "", q: "" };

function isTypingTarget(el: EventTarget | null): boolean {
  if (!(el instanceof HTMLElement)) return false;
  const tag = el.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || el.isContentEditable;
}

export default function Review() {
  const toast = useToast();
  const [filters, setFilters] = useState<Filters>(NO_FILTERS);
  const [items, setItems] = useState<Item[]>([]);
  const [total, setTotal] = useState(0);
  const [pending, setPending] = useState(0);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState(0);
  const [editing, setEditing] = useState(false);
  const [reviewer, setReviewer] = useState(() => {
    try {
      return window.localStorage.getItem(REVIEWER_KEY) || "varun";
    } catch {
      return "varun";
    }
  });
  const rowRefs = useRef<Map<string, HTMLDivElement>>(new Map());

  useEffect(() => {
    try {
      window.localStorage.setItem(REVIEWER_KEY, reviewer);
    } catch {
      // storage unavailable, keep in memory
    }
  }, [reviewer]);

  // debounce the search box, apply other filters immediately
  const [q, setQ] = useState("");
  useEffect(() => {
    const id = window.setTimeout(() => setFilters((f) => ({ ...f, q })), 250);
    return () => window.clearTimeout(id);
  }, [q]);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const r = await getItems({ ...filters, limit: 500 });
      setItems(r.items);
      setTotal(r.total);
      setPending(r.counts.pending);
      setSelected((s) => Math.min(s, Math.max(0, r.items.length - 1)));
    } catch (e) {
      toast.error((e as ApiError).message);
    } finally {
      setLoading(false);
    }
  }, [filters, toast]);

  useEffect(() => {
    void load();
  }, [load]);

  const current = items[selected];

  const applyPatch = useCallback(
    async (item: Item, patch: Omit<ItemPatch, "reviewed_by">, okMsg: string) => {
      try {
        const updated = await patchItem(item.id, { ...patch, reviewed_by: reviewer });
        setItems((prev) => prev.map((it) => (it.id === updated.id ? updated : it)));
        setPending((p) => {
          const was = item.review_status === "pending" ? 1 : 0;
          const now = updated.review_status === "pending" ? 1 : 0;
          return p - was + now;
        });
        toast.success(okMsg);
        setEditing(false);
      } catch (e) {
        const err = e as ApiError;
        toast.error(err.status === 400 ? err.message : `patch failed: ${err.message}`);
      }
    },
    [reviewer, toast],
  );

  const setStatus = useCallback(
    (status: ReviewStatus) => {
      if (!current) return;
      void applyPatch(current, { review_status: status }, `${current.id} ${status}`);
    },
    [current, applyPatch],
  );

  // keyboard shortcuts
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (editing || isTypingTarget(e.target) || e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key === "j") {
        e.preventDefault();
        setSelected((s) => Math.min(items.length - 1, s + 1));
      } else if (e.key === "k") {
        e.preventDefault();
        setSelected((s) => Math.max(0, s - 1));
      } else if (e.key === "a") {
        e.preventDefault();
        setStatus("approved");
      } else if (e.key === "r") {
        e.preventDefault();
        setStatus("rejected");
      } else if (e.key === "e") {
        e.preventDefault();
        setEditing(true);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [editing, items.length, setStatus]);

  useEffect(() => {
    if (!current) return;
    rowRefs.current.get(current.id)?.scrollIntoView({ block: "nearest" });
  }, [current]);

  useEffect(() => setEditing(false), [selected]);

  const set = (k: keyof Filters) => (v: string) => setFilters((f) => ({ ...f, [k]: v }));

  return (
    <div>
      <PageHeader
        title="Dataset review"
        subtitle={
          <>
            <span className="mono text-text">{pending}</span> pending across the whole set,{" "}
            <span className="mono text-text">{total}</span> matching the current filters. Shortcuts:{" "}
            <Kbd>j</Kbd> <Kbd>k</Kbd> move, <Kbd>a</Kbd> approve, <Kbd>r</Kbd> reject, <Kbd>e</Kbd> edit.
          </>
        }
        right={
          <label className="flex items-center gap-2 text-[12px] text-muted">
            reviewer
            <input className="input w-[140px] !py-1" value={reviewer} onChange={(e) => setReviewer(e.target.value)} />
          </label>
        }
      />

      <div className="flex flex-wrap items-center gap-2 mb-4">
        <Select value={filters.status} onChange={set("status")} placeholder="any status" options={STATUSES} />
        <Select value={filters.topic} onChange={set("topic")} placeholder="any topic" options={TOPICS} />
        <Select value={filters.label} onChange={set("label")} placeholder="any label" options={LABELS} />
        <Select value={filters.source} onChange={set("source")} placeholder="any source" options={SOURCES} />
        <input
          className="input flex-1 min-w-[200px]"
          placeholder="search turns"
          value={q}
          onChange={(e) => setQ(e.target.value)}
        />
        {(filters.status || filters.topic || filters.label || filters.source || q) && (
          <button
            type="button"
            className="btn-ghost"
            onClick={() => {
              setFilters(NO_FILTERS);
              setQ("");
            }}
          >
            reset
          </button>
        )}
        {loading && <Spinner />}
      </div>

      {!loading && items.length === 0 ? (
        <Empty title="no items match">
          Loosen the filters, or run <span className="mono text-text">halluscope build-data</span> to create the
          augmented set.
        </Empty>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-[minmax(320px,2fr)_3fr] gap-5 items-start">
          <div className="panel overflow-hidden md:max-h-[calc(100vh-240px)] md:overflow-y-auto">
            {items.map((it, i) => (
              <div
                key={it.id}
                ref={(el) => {
                  if (el) rowRefs.current.set(it.id, el);
                  else rowRefs.current.delete(it.id);
                }}
                role="button"
                tabIndex={0}
                onClick={() => setSelected(i)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") setSelected(i);
                }}
                className={`relative px-4 py-3 border-b border-line/70 last:border-b-0 cursor-pointer transition ${
                  i === selected ? "bg-panel-2" : "hover:bg-panel-2/60"
                }`}
              >
                {i === selected && <span className="absolute left-0 top-0 bottom-0 w-[2px] bg-safe" />}
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="mono text-[12px] text-text">{it.id}</span>
                  <LabelChip label={it.label} />
                  <span className="text-[11px] text-muted">{it.topic}</span>
                  <span className="ml-auto">
                    <StatusChip status={it.review_status} />
                  </span>
                </div>
                <div className="mt-1 text-[12.5px] text-muted line-clamp-2 leading-5">
                  {lastUser(it.turns)}
                </div>
              </div>
            ))}
          </div>

          <div className="min-w-0 md:sticky md:top-[72px]">
            {current ? (
              <ItemDetail
                key={current.id}
                item={current}
                editing={editing}
                onEdit={() => setEditing(true)}
                onCancel={() => setEditing(false)}
                onApprove={() => setStatus("approved")}
                onReject={() => setStatus("rejected")}
                onSave={(patch) => applyPatch(current, patch, `${current.id} saved`)}
              />
            ) : (
              <Panel>
                <Spinner />
              </Panel>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function lastUser(turns: Turn[]): string {
  for (let i = turns.length - 1; i >= 0; i--) {
    if (turns[i].role === "user") return turns[i].content;
  }
  return turns[turns.length - 1]?.content ?? "";
}

function Kbd({ children }: { children: string }) {
  return <span className="kbd">{children}</span>;
}

function Select({
  value,
  onChange,
  placeholder,
  options,
}: {
  value: string;
  onChange: (v: string) => void;
  placeholder: string;
  options: string[];
}) {
  return (
    <select className="select" value={value} onChange={(e) => onChange(e.target.value)}>
      <option value="">{placeholder}</option>
      {options.map((o) => (
        <option key={o} value={o}>
          {o}
        </option>
      ))}
    </select>
  );
}

interface Draft {
  turns: Turn[];
  gap: string;
  expected_clarifying_question: string;
  plausible_silent_assumption: string;
}

function ItemDetail({
  item,
  editing,
  onEdit,
  onCancel,
  onApprove,
  onReject,
  onSave,
}: {
  item: Item;
  editing: boolean;
  onEdit: () => void;
  onCancel: () => void;
  onApprove: () => void;
  onReject: () => void;
  onSave: (patch: Omit<ItemPatch, "reviewed_by">) => void;
}) {
  const initial = useMemo<Draft>(
    () => ({
      turns: item.turns.map((t) => ({ ...t })),
      gap: item.gap ?? "",
      expected_clarifying_question: item.expected_clarifying_question ?? "",
      plausible_silent_assumption: item.plausible_silent_assumption ?? "",
    }),
    [item],
  );
  const [draft, setDraft] = useState<Draft>(initial);
  useEffect(() => setDraft(initial), [initial, editing]);

  const save = () => {
    const patch: Omit<ItemPatch, "reviewed_by"> = {};
    if (JSON.stringify(draft.turns) !== JSON.stringify(initial.turns)) patch.turns = draft.turns;
    if (draft.gap !== initial.gap) patch.gap = draft.gap || null;
    if (draft.expected_clarifying_question !== initial.expected_clarifying_question)
      patch.expected_clarifying_question = draft.expected_clarifying_question || null;
    if (draft.plausible_silent_assumption !== initial.plausible_silent_assumption)
      patch.plausible_silent_assumption = draft.plausible_silent_assumption || null;
    if (Object.keys(patch).length === 0) {
      onCancel();
      return;
    }
    patch.review_status = "edited";
    onSave(patch);
  };

  return (
    <Panel className="animate-rise">
      <div className="flex flex-wrap items-center gap-2 mb-4">
        <span className="mono text-[14px] text-text">{item.id}</span>
        <LabelChip label={item.label} />
        <StatusChip status={item.review_status} />
        <span className="text-[11px] text-muted">
          {item.topic} {"·"} family <span className="mono">{item.family}</span> {"·"} variant{" "}
          <span className="mono">{item.variant}</span> {"·"} {item.source}
        </span>
        {item.reviewed_by && <span className="text-[11px] text-dim">reviewed by {item.reviewed_by}</span>}
        <div className="ml-auto flex items-center gap-1.5">
          {editing ? (
            <>
              <button type="button" className="btn-primary" onClick={save}>
                save
              </button>
              <button type="button" className="btn-ghost" onClick={onCancel}>
                cancel
              </button>
            </>
          ) : (
            <>
              <button type="button" className="btn-primary" onClick={onApprove} title="a">
                approve
              </button>
              <button type="button" className="btn-risk" onClick={onReject} title="r">
                reject
              </button>
              <button type="button" className="btn-ghost" onClick={onEdit} title="e">
                edit
              </button>
            </>
          )}
        </div>
      </div>

      <div className="label mb-2">turns</div>
      <div className="flex flex-col gap-2 mb-5">
        {draft.turns.map((t, i) => (
          <div
            key={i}
            className={`rounded-md border px-3 py-2 ${
              t.role === "user" ? "border-line bg-ink/60" : "border-incons/20 bg-[#141626]"
            }`}
          >
            <div className="flex items-center gap-2 mb-1">
              <span className="mono text-[10px] text-dim">{String(i + 1).padStart(2, "0")}</span>
              <RoleTag role={t.role} />
            </div>
            {editing ? (
              <textarea
                className="w-full bg-transparent text-[13px] leading-5 text-text outline-none resize-y min-h-[48px]"
                value={t.content}
                onChange={(e) =>
                  setDraft((d) => ({
                    ...d,
                    turns: d.turns.map((x, j) => (j === i ? { ...x, content: e.target.value } : x)),
                  }))
                }
              />
            ) : (
              <div className="text-[13px] leading-5 text-text whitespace-pre-wrap">{t.content}</div>
            )}
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <Field
          label="gap"
          tone="risk"
          value={draft.gap}
          editing={editing}
          onChange={(v) => setDraft((d) => ({ ...d, gap: v }))}
        />
        <Field
          label="expected clarifying question"
          tone="safe"
          value={draft.expected_clarifying_question}
          editing={editing}
          onChange={(v) => setDraft((d) => ({ ...d, expected_clarifying_question: v }))}
        />
        <Field
          label="plausible silent assumption"
          tone="incons"
          value={draft.plausible_silent_assumption}
          editing={editing}
          onChange={(v) => setDraft((d) => ({ ...d, plausible_silent_assumption: v }))}
        />
      </div>
      {item.reference_specified_variant && (
        <div className="mt-4 text-[11px] text-dim">
          reference specified variant <span className="mono text-muted">{item.reference_specified_variant}</span>
        </div>
      )}
    </Panel>
  );
}

function Field({
  label,
  value,
  editing,
  onChange,
  tone,
}: {
  label: string;
  value: string;
  editing: boolean;
  onChange: (v: string) => void;
  tone: "risk" | "safe" | "incons";
}) {
  const bar = tone === "risk" ? "bg-risk" : tone === "safe" ? "bg-safe" : "bg-incons";
  return (
    <div className="relative pl-3">
      <span className={`absolute left-0 top-0 bottom-0 w-[2px] rounded ${bar} opacity-70`} />
      <div className="label mb-1">{label}</div>
      {editing ? (
        <textarea
          className="input w-full min-h-[72px] resize-y text-[12.5px] leading-5"
          value={value}
          onChange={(e) => onChange(e.target.value)}
        />
      ) : (
        <div className="text-[12.5px] leading-5 text-text/90">
          {value || <span className="text-dim">none</span>}
        </div>
      )}
    </div>
  );
}
