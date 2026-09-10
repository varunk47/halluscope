import { createContext, useCallback, useContext, useMemo, useRef, useState, type ReactNode } from "react";

export type ToastTone = "error" | "info" | "success";

interface ToastItem {
  id: number;
  tone: ToastTone;
  text: string;
}

interface ToastApi {
  push: (text: string, tone?: ToastTone) => void;
  error: (text: string) => void;
  info: (text: string) => void;
  success: (text: string) => void;
}

const ToastCtx = createContext<ToastApi | null>(null);

const TONE_CLASS: Record<ToastTone, string> = {
  error: "border-risk-hot/50 bg-[#1a1114] text-text",
  info: "border-safe/40 bg-[#0f1a1f] text-text",
  success: "border-safe/50 bg-[#0f1a1f] text-text",
};

const TONE_DOT: Record<ToastTone, string> = {
  error: "bg-risk-hot",
  info: "bg-safe",
  success: "bg-safe",
};

export function ToastProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<ToastItem[]>([]);
  const counter = useRef(0);

  const remove = useCallback((id: number) => {
    setItems((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const push = useCallback(
    (text: string, tone: ToastTone = "info") => {
      const id = ++counter.current;
      setItems((prev) => [...prev, { id, tone, text }]);
      window.setTimeout(() => remove(id), tone === "error" ? 7000 : 3500);
    },
    [remove],
  );

  const api = useMemo<ToastApi>(
    () => ({
      push,
      error: (t) => push(t, "error"),
      info: (t) => push(t, "info"),
      success: (t) => push(t, "success"),
    }),
    [push],
  );

  return (
    <ToastCtx.Provider value={api}>
      {children}
      <div className="fixed bottom-5 right-5 z-50 flex flex-col gap-2 w-[min(420px,calc(100vw-40px))]">
        {items.map((t) => (
          <div
            key={t.id}
            role="status"
            className={`animate-pop flex items-start gap-3 rounded-lg border px-3 py-2.5 text-[13px] shadow-panel ${TONE_CLASS[t.tone]}`}
          >
            <span className={`mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full ${TONE_DOT[t.tone]}`} />
            <span className="flex-1 leading-5 break-words">{t.text}</span>
            <button
              type="button"
              onClick={() => remove(t.id)}
              className="text-muted hover:text-text transition text-[12px] leading-5"
              aria-label="dismiss"
            >
              close
            </button>
          </div>
        ))}
      </div>
    </ToastCtx.Provider>
  );
}

export function useToast(): ToastApi {
  const ctx = useContext(ToastCtx);
  if (!ctx) throw new Error("useToast must be used inside ToastProvider");
  return ctx;
}
