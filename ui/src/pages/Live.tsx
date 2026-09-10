import { useCallback, useEffect, useRef, useState } from "react";
import {
  ApiError,
  chatStream,
  getItems,
  scoreTurns,
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
}

const EMPTY_STREAM: StreamState = { active: false, mode: null, gate: null, text: "" };

export default function Live() {
  const toast = useToast();
  const page = useRef<HTMLDivElement>(null);
  const [turns, setTurns] = useState<DialogTurn[]>([{ role: "user", content: "" }]);
  const [presets, setPresets] = useState<Item[]>([]);
  const [presetId, setPresetId] = useState("");
  const [score, setScore] = useState<ScoreResponse | null>(null);
  const [scoring, setScoring] = useState(false);
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
    getItems({ limit: 64, source: "seed" })
      .then((r) => setPresets(r.items))
      .catch((e: ApiError) => toast.error(`presets: ${e.message}`));
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
    setStream({ active: true, mode: force ?? null, gate: null, text: "" });
    let gate: GateEvent | null = null;
    try {
      await chatStream(
        { turns: stripMeta(cleaned), force },
        {
          onGate: (g) => {
            gate = g;
            setStream((s) => ({ ...s, gate: g, mode: g.mode }));
          },
          onToken: (t) => setStream((s) => ({ ...s, text: s.text + t.text })),
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
            <button type="button" className="btn-primary" onClick={onScore} disabled={busy}>
              {scoring ? <Spinner label="scoring" /> : "Score"}
            </button>
            <button type="button" className="btn-risk" onClick={() => onChat()} disabled={busy}>
              {stream.active ? <Spinner label="streaming" /> : "Ask or answer"}
            </button>
            <div className="flex items-center gap-1 pl-2 border-l border-line">
              <button type="button" className="btn-ghost btn-xs" onClick={() => onChat("ask")} disabled={busy}>
                force ask
              </button>
              <button type="button" className="btn-ghost btn-xs" onClick={() => onChat("answer")} disabled={busy}>
                force answer
              </button>
              {stream.active && (
                <button type="button" className="btn-ghost btn-xs" onClick={() => abortRef.current?.abort()}>
                  stop
                </button>
              )}
            </div>
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
          {score ? <ScoreView score={score} /> : <Idle />}
        </div>
      </div>
    </div>
  );
}

function Idle() {
  return (
    <Panel className="grid-texture">
      <SectionLabel>readout</SectionLabel>
      <p className="text-[13.5px] text-muted leading-6">
        Nothing scored yet. Score reads the hidden state at the position where the assistant would start
        answering and reports, layer by layer, how strongly it looks like an underspecified request. The
        figure at the top is the best validation layer; the strip below it is the whole sweep.
      </p>
      <ul className="mt-4 grid grid-cols-3 gap-3 text-[12px] text-dim">
        <li>
          <span className="block text-safe mono text-[13px]">0.0</span>
          fully specified
        </li>
        <li>
          <span className="block text-muted mono text-[13px]">0.5</span>
          probe boundary
        </li>
        <li>
          <span className="block text-risk mono text-[13px]">1.0</span>
          missing a needed detail
        </li>
      </ul>
    </Panel>
  );
}

function ScoreView({ score }: { score: ScoreResponse }) {
  const uqEntries = Object.entries(score.uq);
  return (
    <>
      <RiskRibbon score={score} />
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
          <span className="chip border-line-2 text-muted">
            <span className="h-1.5 w-1.5 rounded-full bg-safe animate-pulseDot" />
            reading the gate
          </span>
        )}
      </div>
      <div className="px-3 pb-3 pt-1.5 text-[13.5px] leading-6 text-text whitespace-pre-wrap min-h-[40px]">
        {stream.text}
        <span className="inline-block w-[7px] h-[14px] align-[-2px] ml-0.5 bg-safe/80 animate-pulseDot" />
      </div>
    </div>
  );
}
