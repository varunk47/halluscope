import { useCallback, useEffect, useRef, useState } from "react";
import {
  ApiError,
  chatStream,
  getItems,
  getResult,
  scoreTurns,
  type ProbeFull,
  type GateEvent,
  type Item,
  type ScoreResponse,
} from "../api";
import DialogueBuilder, { MetaBadge, stripMeta, type DialogTurn, type TurnMeta } from "../components/DialogueBuilder";
import { PageHeader, Panel, SectionLabel, Spinner } from "../components/Primitives";
import RiskRibbon from "../components/RiskRibbon";
import LayerBars from "../components/charts/LayerBars";
import TurnTrajectory from "../components/charts/TurnTrajectory";
import EntropyLine from "../components/charts/EntropyLine";
import { useToast } from "../components/Toast";
import { gsap, reducedMotion, useGSAP } from "../lib/motion";

interface StreamState {
  active: boolean;
  mode: "ask" | "answer" | null;
  gate: GateEvent | null;
  text: string;
  chunks: string[];
}

const EMPTY_STREAM: StreamState = { active: false, mode: null, gate: null, text: "", chunks: [] };
const OPENING_PRESET = "rag-0004-d";
const REFERENCE_RESULT = "probe_qwen_gap_last_minimal";

export default function Live() {
  const toast = useToast();
  const page = useRef<HTMLDivElement>(null);
  const [turns, setTurns] = useState<DialogTurn[]>([{ role: "user", content: "" }]);
  const [presets, setPresets] = useState<Item[]>([]);
  const [reference, setReference] = useState<ProbeFull | null>(null);
  const [presetId, setPresetId] = useState("");
  const [score, setScore] = useState<ScoreResponse | null>(null);
  const [scoring, setScoring] = useState(false);
  const [justScored, setJustScored] = useState(false);
  const [stream, setStream] = useState<StreamState>(EMPTY_STREAM);
  const abortRef = useRef<AbortController | null>(null);

  // One reveal on arrival, nothing after: the bench settles into place once.
  useGSAP(
    () => {
      if (reducedMotion()) return;
      gsap.from("[data-reveal]", { y: 10, autoAlpha: 0, duration: 0.5, stagger: 0.08, ease: "power2.out" });
    },
    { scope: page },
  );

  useEffect(() => {
    getItems({ limit: 256, source: "seed" })
      .then((r) => {
        setPresets(r.items);
        // Arrive with a request on the bench: a multi-turn contradiction, the
        // case the whole project is about, rather than an empty box.
        const first = r.items.find((p) => p.id === OPENING_PRESET) ?? r.items.find((p) => p.variant === "d");
        if (first) {
          setPresetId(first.id);
          setTurns(first.turns.map((t) => ({ ...t })));
        }
      })
      .catch((e: ApiError) => toast.error(`presets: ${e.message}`));
    getResult<ProbeFull>(REFERENCE_RESULT)
      .then(setReference)
      .catch(() => setReference(null));
    return () => abortRef.current?.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const loadPreset = (id: string) => {
    setPresetId(id);
    const item = presets.find((p) => p.id === id);
    if (!item) return;
    setTurns(item.turns.map((t) => ({ ...t })));
    setScore(null);
    setStream(EMPTY_STREAM);
  };

  const validate = useCallback((): DialogTurn[] | null => {
    const cleaned = turns.filter((t) => t.content.trim().length > 0);
    if (cleaned.length === 0) {
      toast.error("add at least one user turn before scoring");
      return null;
    }
    if (cleaned[cleaned.length - 1].role !== "user") {
      toast.error("dialogue must end on a user turn");
      return null;
    }
    return cleaned;
  }, [turns, toast]);

  const onScore = async () => {
    const cleaned = validate();
    if (!cleaned) return;
    setScoring(true);
    try {
      const res = await scoreTurns(stripMeta(cleaned), true);
      setScore(res);
      setJustScored(true);
      window.setTimeout(() => setJustScored(false), 1400);
    } catch (e) {
      toast.error((e as ApiError).message);
    } finally {
      setScoring(false);
    }
  };

  const onChat = async (force?: "ask" | "answer") => {
    const cleaned = validate();
    if (!cleaned) return;
    abortRef.current?.abort();
    const ac = new AbortController();
    abortRef.current = ac;
    setStream({ active: true, mode: force ?? null, gate: null, text: "", chunks: [] });
    let gate: GateEvent | null = null;
    try {
      await chatStream(
        { turns: stripMeta(cleaned), force },
        {
          onGate: (g) => {
            gate = g;
            setStream((s) => ({ ...s, gate: g, mode: g.mode }));
          },
          onToken: (t) => setStream((s) => ({ ...s, text: s.text + t.text, chunks: [...s.chunks, t.text] })),
          onDone: (d) => {
            const meta: TurnMeta = { mode: d.mode, gate };
            setTurns([
              ...cleaned,
              { role: "assistant", content: d.text.trim(), meta },
              { role: "user", content: "" },
            ]);
            setStream(EMPTY_STREAM);
          },
        },
        ac.signal,
      );
    } catch (e) {
      const aborted = ac.signal.aborted || (e as Error).name === "AbortError";
      if (!aborted) toast.error((e as ApiError).message);
    } finally {
      // whatever happened, never leave the page stuck in the streaming state
      setStream(EMPTY_STREAM);
    }
  };

  const busy = scoring || stream.active;

  return (
    <div ref={page}>
      <PageHeader
        title="Read the model before it answers"
        subtitle="Score reads the probe at every layer of the residual stream. Ask or answer lets the gate decide whether to clarify first."
        right={
          <>
            <button type="button" className="btn-primary min-w-[88px] justify-center" onClick={onScore} disabled={busy}>
              {scoring ? <span className="shimmer-text">scoring</span> : justScored ? <Check /> : "Score"}
            </button>
            <button type="button" className="btn-risk" onClick={() => onChat()} disabled={busy}>
              {stream.active ? <Spinner label="streaming" /> : "Ask or answer"}
            </button>
            {stream.active && (
              <button type="button" className="btn-ghost" onClick={() => abortRef.current?.abort()}>
                stop
              </button>
            )}
          </>
        }
      />

      <div className="grid grid-cols-1 md:grid-cols-[3fr_2fr] gap-5 items-start">
        <div className="min-w-0" data-reveal="dialogue">
          <SectionLabel right={`${turns.length} turn${turns.length === 1 ? "" : "s"}`}>dialogue</SectionLabel>
          <DialogueBuilder
            turns={turns}
            onChange={(next) => {
              setTurns(next);
              setPresetId("");
            }}
            presets={presets}
            presetId={presetId}
            onPreset={loadPreset}
            disabled={busy}
            trailing={stream.active ? <StreamBubble stream={stream} /> : null}
          />
        </div>

        <div className="min-w-0 flex flex-col gap-4 md:sticky md:top-[72px]" data-reveal="readout">
          {scoring && !score ? (
            <Scanning />
          ) : score ? (
            <ScoreView score={score} onForce={onChat} busy={busy} />
          ) : (
            <Idle reference={reference} />
          )}
        </div>
      </div>
    </div>
  );
}

/** The strip's shape, before the strip: a light passes over where the bars will be. */
function Scanning() {
  return (
    <>
      <div className="panel p-5">
        <div className="skel h-3 w-40" />
        <div className="skel h-16 w-52 mt-3" />
        <div className="skel h-2 w-full mt-6 rounded-full" />
      </div>
      <div className="panel p-5 pb-8">
        <div className="skel h-3 w-56 mb-4" />
        <div className="relative h-[150px] overflow-hidden rounded-lg">
          <div className="absolute inset-0 flex items-end gap-[3px]">
            {Array.from({ length: 33 }, (_, i) => (
              <div key={i} className="flex-1 rounded-t-[2px] bg-panel-2" style={{ height: `${18 + ((i * 37) % 40)}%` }} />
            ))}
          </div>
          <div className="absolute inset-y-0 w-1/3 animate-scan bg-gradient-to-r from-transparent via-safe/10 to-transparent" />
        </div>
      </div>
      <div className="text-[12px] shimmer-text">reading 33 layers at the answer position</div>
    </>
  );
}

function Check() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" aria-label="scored" className="animate-pop">
      <path d="M3 8.5l3 3 7-7" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

/**
 * Before anything is scored the column still shows a reading: the linear
 * probe's validation AUROC at every layer of the model, from the results
 * file. It is the same strip the live score will draw, so the eye already
 * knows where to look when the number lands.
 */
function Idle({ reference }: { reference: ProbeFull | null }) {
  const linear = reference?.probes?.linear;
  const curve = linear?.per_layer_val_auroc ?? {};
  const layers = Object.keys(curve)
    .map(Number)
    .sort((a, b) => a - b)
    .map((l) => ({ layer: l, prob: curve[String(l)], decision: curve[String(l)] }));
  const best = linear?.best_layer ?? null;
  const bestAuroc = best !== null ? curve[String(best)] : undefined;
  return (
    <>
      <div className="relative overflow-hidden rounded-xl border border-line bg-panel shadow-readout px-5 pt-5 pb-4">
        <div className="label">the probe on the test set, before your request</div>
        <div className="flex items-baseline gap-1.5 mt-1">
          <span className="font-display mono text-[64px] leading-none font-semibold tracking-[-0.03em] text-text">
            {bestAuroc !== undefined ? bestAuroc.toFixed(3) : "\u2014".replace("\u2014", "")}
          </span>
          {bestAuroc !== undefined && <span className="text-[13px] text-muted mb-2">validation AUROC at layer {best}</span>}
        </div>
        <p className="mt-3 text-[13px] text-muted leading-5 max-w-[46ch]">
          Press Score and this face becomes the reading for the dialogue on the left: how strongly the hidden
          state says a needed detail is missing.
        </p>
      </div>
      <Panel className="pb-8">
        {layers.length > 0 ? (
          <LayerBars
            data={layers}
            bestLayer={best}
            title="linear probe, validation AUROC by layer"
            format={(v) => v.toFixed(3)}
            floor={0.5}
            color={() => "#22d3ee"}
          />
        ) : (
          <div className="text-[12px] text-dim">no probe result on disk yet; run halluscope probe</div>
        )}
      </Panel>
    </>
  );
}

function ScoreView({
  score,
  onForce,
  busy,
}: {
  score: ScoreResponse;
  onForce: (mode: "ask" | "answer") => void;
  busy: boolean;
}) {
  const uqEntries = Object.entries(score.uq);
  return (
    <>
      <RiskRibbon score={score} />
      <div className="flex items-center gap-2 -mt-1 text-[12.5px] text-muted">
        <span>try the other branch:</span>
        <button type="button" className="btn-ghost btn-xs" onClick={() => onForce("ask")} disabled={busy}>
          make it ask
        </button>
        <button type="button" className="btn-ghost btn-xs" onClick={() => onForce("answer")} disabled={busy}>
          make it answer
        </button>
      </div>
      <Panel className="pb-8">
        <LayerBars data={score.per_layer} bestLayer={score.best_layer} />
      </Panel>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <Panel>
          <TurnTrajectory probs={score.per_turn_prob_best} />
        </Panel>
        <Panel>
          <EntropyLine values={score.logit_lens_entropy} />
        </Panel>
      </div>
      <Panel>
        <SectionLabel right={`${uqEntries.length} signal${uqEntries.length === 1 ? "" : "s"}`}>
          uncertainty signals on the same prefix
        </SectionLabel>
        {uqEntries.length === 0 ? (
          <div className="text-[12px] text-dim">no uncertainty signals returned</div>
        ) : (
          <div className="grid grid-cols-2 gap-x-6 gap-y-2">
            {uqEntries.map(([k, v]) => (
              <div key={k} className="flex items-baseline justify-between gap-3 border-b border-line/70 pb-1.5">
                <span className="text-[12px] text-muted truncate" title={k}>
                  {k.replace(/_/g, " ")}
                </span>
                <span className="mono text-[13px] text-text">{Number.isFinite(v) ? v.toFixed(3) : "n/a"}</span>
              </div>
            ))}
          </div>
        )}
      </Panel>
    </>
  );
}

function StreamBubble({ stream }: { stream: StreamState }) {
  const meta: TurnMeta | null = stream.mode ? { mode: stream.mode, gate: stream.gate } : null;
  return (
    <div className="animate-rise rounded-xl border border-incons/30 bg-[#131626]">
      <div className="flex items-center gap-2 px-3 pt-2">
        <span className="text-[12px] font-medium text-incons">assistant</span>
        {meta ? (
          <MetaBadge meta={meta} />
        ) : (
          <span className="chip border-line-2 shimmer-text">reading the gate</span>
        )}
      </div>
      <div className="px-3 pb-3 pt-1.5 text-[13.5px] leading-6 text-text whitespace-pre-wrap min-h-[40px]">
        {stream.chunks.map((c, i) => (
          <span key={i} className="animate-tokenIn">
            {c}
          </span>
        ))}
        <span className="inline-block w-[7px] h-[14px] align-[-2px] ml-0.5 bg-safe/80 animate-pulseDot" />
      </div>
    </div>
  );
}
