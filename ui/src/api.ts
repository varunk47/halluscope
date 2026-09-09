// Typed fetch helpers for the HalluScope FastAPI backend.
// All paths are relative (/api/...) so the Vite proxy and the built bundle
// served by FastAPI both work without configuration.

export type Role = "user" | "assistant";
export interface Turn {
  role: Role;
  content: string;
}

export type Label = "specified" | "underspecified" | "inconsistent";
export type ReviewStatus = "pending" | "approved" | "edited" | "rejected";
export type Source = "seed" | "augmented" | "public";
export type Variant = "a" | "b" | "c" | "d";

export interface Item {
  id: string;
  family: string;
  topic: string;
  variant: Variant;
  label: Label;
  turns: Turn[];
  gap: string | null;
  expected_clarifying_question: string | null;
  plausible_silent_assumption: string | null;
  reference_specified_variant: string | null;
  source: Source;
  review_status: ReviewStatus;
  reviewed_by: string | null;
}

export interface ItemsResponse {
  total: number;
  counts: { pending: number };
  items: Item[];
}

export interface ItemsQuery {
  status?: string;
  topic?: string;
  label?: string;
  source?: string;
  q?: string;
  limit?: number;
  offset?: number;
}

export interface ItemPatch {
  review_status?: ReviewStatus;
  turns?: Turn[];
  gap?: string | null;
  expected_clarifying_question?: string | null;
  plausible_silent_assumption?: string | null;
  reviewed_by: string;
}

export interface LayerScore {
  layer: number;
  prob: number;
  decision: number;
}

export interface ScoreResponse {
  model: string;
  best_layer: number | null;
  gate_fired: boolean | null;
  threshold: number | null;
  prob_best: number | null;
  per_layer: LayerScore[];
  per_turn_prob_best: number[];
  logit_lens_entropy: number[];
  uq: Record<string, number>;
  seconds: number;
}

export interface GateInfo {
  fitted: boolean;
  layer?: number;
  threshold?: number;
  alpha?: number;
  reason?: string;
  n_train?: number;
  n_val?: number;
  val_auroc_best?: number;
  [k: string]: unknown;
}

export interface Health {
  ok: boolean;
  version: string;
  model_loaded: string | null;
  gate: GateInfo;
}

export interface Metric {
  n: number;
  auroc: number;
  auroc_lo: number;
  auroc_hi: number;
  auprc: number;
  auprc_lo: number;
  auprc_hi: number;
  accuracy?: number;
  ece?: number;
  reliability?: Reliability;
}

export interface Reliability {
  confidence: number[];
  accuracy: number[];
  count: number[];
}

export type ResultKind = "probe" | "uq" | "behavior" | "loop";

export interface ProbeSummaryEntry {
  best_layer: number;
  test: Metric;
}

export interface LotoEntry {
  auroc: number;
  auroc_lo: number;
  auroc_hi: number;
  n?: number;
}

export interface ResultSummary {
  name: string;
  kind: ResultKind | string;
  model: string | null;
  // probe
  target?: string;
  pooling?: string;
  n_layers?: number;
  split_sizes?: Record<string, number>;
  probes?: Record<string, ProbeSummaryEntry>;
  baselines?: Record<string, Metric>;
  loto?: Record<string, LotoEntry>;
  // uq / behavior / loop
  summary?: Record<string, unknown>;
  split?: string | null;
  condition?: string | null;
  seed?: number | null;
  cost?: unknown;
}

export interface ProbeFull {
  model: string;
  target?: string;
  pooling?: string;
  n_layers?: number;
  probes: Record<
    string,
    {
      best_layer: number;
      test: Metric;
      val?: Metric;
      per_layer_val_auroc?: Record<string, number>;
      per_layer_test_auroc?: Record<string, number>;
    }
  >;
  baselines?: Record<string, Metric>;
  loto?: Record<string, LotoEntry>;
  [k: string]: unknown;
}

export interface Provenance {
  python: string;
  platform: string;
  torch: string;
  transformers: string;
  cuda: boolean;
  gpu: string | null;
  seed: number;
  split_fractions: number[];
  n_bootstrap: number;
  gate_alpha: number;
  judge_aliases: Record<string, string[]>;
  llm_calls: Record<string, { calls: number; ok: number; cost_usd: number; models: string[] }>;
  total_cost_usd: number;
}

export class ApiError extends Error {
  status: number;
  constructor(status: number, detail: string) {
    super(detail);
    this.status = status;
    this.name = "ApiError";
  }
}

async function readError(res: Response): Promise<ApiError> {
  let detail = `${res.status} ${res.statusText}`;
  try {
    const body = (await res.json()) as { detail?: unknown };
    if (typeof body.detail === "string") detail = body.detail;
    else if (body.detail) detail = JSON.stringify(body.detail);
  } catch {
    // body was not JSON, keep the status text
  }
  return new ApiError(res.status, detail);
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, init);
  } catch (e) {
    throw new ApiError(0, `backend unreachable (${(e as Error).message})`);
  }
  if (!res.ok) throw await readError(res);
  return (await res.json()) as T;
}

function json(body: unknown, method = "POST"): RequestInit {
  return {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
}

function qs(params: Record<string, string | number | undefined>): string {
  const p = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => {
    if (v !== undefined && v !== "") p.set(k, String(v));
  });
  const s = p.toString();
  return s ? `?${s}` : "";
}

export const getHealth = () => request<Health>("/api/health");

export const getItems = (q: ItemsQuery = {}) =>
  request<ItemsResponse>(`/api/items${qs(q as Record<string, string | number | undefined>)}`);

export const patchItem = (id: string, patch: ItemPatch) =>
  request<Item>(`/api/items/${encodeURIComponent(id)}`, json(patch, "PATCH"));

export const getResults = () => request<{ results: ResultSummary[] }>("/api/results");

export const getResult = <T = Record<string, unknown>>(name: string) =>
  request<T>(`/api/results/${encodeURIComponent(name)}`);

export const getProvenance = () => request<Provenance>("/api/provenance");

export const scoreTurns = (turns: Turn[], uq = true) =>
  request<ScoreResponse>("/api/score", json({ turns, uq }));

// ---- SSE over POST -------------------------------------------------------

export interface GateEvent {
  fired: boolean;
  prob: number | null;
  threshold: number | null;
  mode: "ask" | "answer";
}
export interface TokenEvent {
  text: string;
}
export interface DoneEvent {
  mode: "ask" | "answer";
  text: string;
}

export interface ChatHandlers {
  onGate?: (g: GateEvent) => void;
  onToken?: (t: TokenEvent) => void;
  onDone?: (d: DoneEvent) => void;
  onError?: (err: ApiError) => void;
}

export interface ChatRequest {
  turns: Turn[];
  force?: "ask" | "answer";
  max_new_tokens?: number;
}

/**
 * Consume a text/event-stream response from a POST request. EventSource only
 * supports GET, so we read the body with a ReadableStream reader and parse the
 * `event:` and `data:` frames by hand. Frames are separated by a blank line.
 */
export async function chatStream(
  body: ChatRequest,
  handlers: ChatHandlers,
  signal?: AbortSignal,
): Promise<void> {
  let res: Response;
  try {
    res = await fetch("/api/chat", {
      ...json(body),
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      signal,
    });
  } catch (e) {
    // keep abort semantics intact so callers can stay silent on a user stop
    if ((e as Error).name === "AbortError") throw e;
    const err = new ApiError(0, `backend unreachable (${(e as Error).message})`);
    handlers.onError?.(err);
    throw err;
  }
  if (!res.ok || !res.body) {
    const err = await readError(res);
    handlers.onError?.(err);
    throw err;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  let sawDone = false;
  const isObj = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null;
  const isMode = (v: unknown): v is "ask" | "answer" => v === "ask" || v === "answer";

  const dispatch = (event: string, data: string) => {
    if (!data) return;
    let parsed: unknown;
    try {
      parsed = JSON.parse(data);
    } catch {
      handlers.onError?.(new ApiError(0, `malformed ${event} event from stream`));
      return;
    }
    if (event === "gate") {
      if (isObj(parsed) && typeof parsed.fired === "boolean" && isMode(parsed.mode)) {
        handlers.onGate?.({
          fired: parsed.fired,
          prob: typeof parsed.prob === "number" ? parsed.prob : null,
          threshold: typeof parsed.threshold === "number" ? parsed.threshold : null,
          mode: parsed.mode,
        });
      } else handlers.onError?.(new ApiError(0, "gate event had an unexpected shape"));
    } else if (event === "token") {
      if (isObj(parsed) && typeof parsed.text === "string") handlers.onToken?.({ text: parsed.text });
    } else if (event === "done") {
      if (isObj(parsed) && isMode(parsed.mode) && typeof parsed.text === "string") {
        sawDone = true;
        handlers.onDone?.({ mode: parsed.mode, text: parsed.text });
      } else handlers.onError?.(new ApiError(0, "done event had an unexpected shape"));
    } else if (event === "error") {
      const detail = isObj(parsed) && typeof parsed.detail === "string" ? parsed.detail : data;
      handlers.onError?.(new ApiError(0, detail));
    }
  };

  const consumeFrame = (frame: string) => {
    let event = "message";
    const dataLines: string[] = [];
    for (const raw of frame.split("\n")) {
      const line = raw.replace(/\r$/, "");
      if (!line || line.startsWith(":")) continue;
      const idx = line.indexOf(":");
      const field = idx === -1 ? line : line.slice(0, idx);
      let value = idx === -1 ? "" : line.slice(idx + 1);
      if (value.startsWith(" ")) value = value.slice(1);
      if (field === "event") event = value;
      else if (field === "data") dataLines.push(value);
    }
    dispatch(event, dataLines.join("\n"));
  };

  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    // normalise CRLF so a single split works for both line endings
    buffer = buffer.replace(/\r\n/g, "\n");
    let sep = buffer.indexOf("\n\n");
    while (sep !== -1) {
      const frame = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      consumeFrame(frame);
      sep = buffer.indexOf("\n\n");
    }
  }
  if (buffer.trim()) consumeFrame(buffer);
  if (!sawDone) {
    // the server generator raised or the connection dropped before `done`
    const err = new ApiError(0, "stream ended before a done event; check the server log");
    handlers.onError?.(err);
    throw err;
  }
}

// ---- formatting helpers shared by pages ----------------------------------

export const fmt = {
  pct: (v: number | null | undefined, digits = 1) =>
    v === null || v === undefined || Number.isNaN(v) ? "n/a" : `${(v * 100).toFixed(digits)}%`,
  num: (v: number | null | undefined, digits = 3) =>
    v === null || v === undefined || Number.isNaN(v) ? "n/a" : v.toFixed(digits),
  ci: (m: Pick<Metric, "auroc" | "auroc_lo" | "auroc_hi"> | undefined) =>
    m ? `${m.auroc.toFixed(3)} [${m.auroc_lo.toFixed(2)}, ${m.auroc_hi.toFixed(2)}]` : "n/a",
};
