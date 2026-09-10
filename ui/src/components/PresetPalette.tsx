import { useEffect, useMemo, useRef, useState } from "react";
import type { Item } from "../api";
import { LabelChip } from "./Primitives";

/**
 * A command palette over the seed set. Sixty-four dialogues do not fit in a
 * select; here they are searched by id, topic, label, or any word in the
 * final turn, and picked with the keyboard. Opens on the button or on Ctrl+K.
 */
export default function PresetPalette({
  presets,
  value,
  onPick,
  disabled,
}: {
  presets: Item[];
  value: string;
  onPick: (id: string) => void;
  disabled?: boolean;
}) {
  const [open, setOpen] = useState(false);
  const [q, setQ] = useState("");
  const [cursor, setCursor] = useState(0);
  const input = useRef<HTMLInputElement>(null);
  const list = useRef<HTMLUListElement>(null);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        if (!disabled) setOpen((o) => !o);
      }
      if (e.key === "Escape") setOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [disabled]);

  useEffect(() => {
    if (open) {
      setQ("");
      setCursor(0);
      window.setTimeout(() => input.current?.focus(), 0);
    }
  }, [open]);

  const hits = useMemo(() => {
    const words = q.toLowerCase().split(/\s+/).filter(Boolean);
    if (words.length === 0) return presets;
    return presets.filter((p) => {
      const hay = `${p.id} ${p.topic} ${p.label} ${p.turns[p.turns.length - 1]?.content ?? ""}`.toLowerCase();
      return words.every((w) => hay.includes(w));
    });
  }, [presets, q]);

  useEffect(() => {
    setCursor(0);
  }, [q]);

  useEffect(() => {
    list.current?.children[cursor]?.scrollIntoView({ block: "nearest" });
  }, [cursor]);

  const choose = (id: string) => {
    onPick(id);
    setOpen(false);
  };

  const current = presets.find((p) => p.id === value);

  return (
    <>
      <button
        type="button"
        className="input flex-1 min-w-[260px] text-left flex items-center gap-3 hover:border-line-2"
        onClick={() => setOpen(true)}
        disabled={disabled}
        aria-haspopup="dialog"
      >
        {current ? (
          <>
            <span className="mono text-text">{current.id}</span>
            <LabelChip label={current.label} />
            <span className="text-muted truncate">{snippet(current)}</span>
          </>
        ) : (
          <span className="text-dim">load a request from the seed set</span>
        )}
        <span className="ml-auto flex items-center gap-1">
          <kbd className="kbd">ctrl</kbd>
          <kbd className="kbd">k</kbd>
        </span>
      </button>

      {open && (
        <div
          className="fixed inset-0 z-50 flex items-start justify-center pt-[12vh] px-4 bg-ink/70 backdrop-blur-sm"
          onMouseDown={(e) => {
            if (e.target === e.currentTarget) setOpen(false);
          }}
          role="dialog"
          aria-label="pick a seed request"
        >
          <div className="w-full max-w-2xl panel overflow-hidden animate-pop">
            <div className="flex items-center gap-3 px-4 border-b border-line">
              <SearchGlyph />
              <input
                ref={input}
                className="flex-1 bg-transparent py-3.5 text-[14px] text-text placeholder:text-dim outline-none"
                placeholder="search by topic, label, id, or a word from the request"
                value={q}
                onChange={(e) => setQ(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "ArrowDown") {
                    e.preventDefault();
                    setCursor((c) => Math.min(hits.length - 1, c + 1));
                  } else if (e.key === "ArrowUp") {
                    e.preventDefault();
                    setCursor((c) => Math.max(0, c - 1));
                  } else if (e.key === "Enter" && hits[cursor]) {
                    choose(hits[cursor].id);
                  }
                }}
              />
              <span className="text-[11px] text-dim whitespace-nowrap">
                {hits.length} of {presets.length}
              </span>
            </div>
            <ul ref={list} className="max-h-[52vh] overflow-y-auto py-1" role="listbox">
              {hits.length === 0 && <li className="px-4 py-6 text-[13px] text-dim">nothing matches; try a topic like rag or agents</li>}
              {hits.map((p, i) => (
                <li
                  key={p.id}
                  role="option"
                  aria-selected={i === cursor}
                  onMouseEnter={() => setCursor(i)}
                  onClick={() => choose(p.id)}
                  className={`flex items-center gap-3 px-4 py-2.5 cursor-pointer ${
                    i === cursor ? "bg-panel-2" : ""
                  }`}
                >
                  <span className="mono text-[12px] text-text w-[132px] shrink-0">{p.id}</span>
                  <LabelChip label={p.label} />
                  <span className="text-[12.5px] text-muted truncate">{snippet(p)}</span>
                  <span className="ml-auto text-[11px] text-dim shrink-0">{p.topic}</span>
                </li>
              ))}
            </ul>
            <div className="flex items-center gap-4 px-4 py-2 border-t border-line text-[11px] text-dim">
              <span>
                <kbd className="kbd">↑</kbd> <kbd className="kbd">↓</kbd> move
              </span>
              <span>
                <kbd className="kbd">↵</kbd> load
              </span>
              <span>
                <kbd className="kbd">esc</kbd> close
              </span>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

function snippet(item: Item): string {
  const last = item.turns[item.turns.length - 1]?.content ?? "";
  const s = last.replace(/\s+/g, " ").trim();
  return s.length > 72 ? `${s.slice(0, 72)}...` : s;
}

function SearchGlyph() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true" className="text-dim shrink-0">
      <circle cx="7" cy="7" r="4.5" fill="none" stroke="currentColor" strokeWidth="1.5" />
      <path d="M10.5 10.5L14 14" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  );
}
