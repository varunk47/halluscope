import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { ApiError, getResult, getResults, type ProbeFull, type ResultSummary } from "../api";
import { Empty, PageHeader } from "../components/Primitives";
import ProbeView from "../components/results/ProbeView";
import UqView from "../components/results/UqView";
import BehaviorView from "../components/results/BehaviorView";
import LoopView from "../components/results/LoopView";
import { useToast } from "../components/Toast";

const KIND_ORDER = ["probe", "uq", "behavior", "loop"];
const KIND_CMD: Record<string, string> = {
  probe: "halluscope probe",
  uq: "halluscope uq",
  behavior: "halluscope behavior",
  loop: "halluscope loop",
};
const KIND_BLURB: Record<string, string> = {
  probe: "linear, mass-mean and MLP probes on cached activations, with baselines, a layer sweep and leave-one-topic-out",
  uq: "sampling and prompting based uncertainty signals compared on the same split",
  behavior: "what the model does unprompted: asks, flags, or silently assumes, judged cross-family",
  loop: "the ask-or-answer loop end to end, graded per condition",
};

const KIND_TONE: Record<string, string> = {
  probe: "text-safe border-safe/40 bg-safe/10",
  uq: "text-incons border-incons/40 bg-incons/10",
  behavior: "text-risk border-risk/40 bg-risk/10",
  loop: "text-text border-line-2 bg-panel-2",
};

export default function Results() {
  const toast = useToast();
  const [params, setParams] = useSearchParams();
  const [list, setList] = useState<ResultSummary[] | null>(null);
  const [full, setFull] = useState<ProbeFull | null>(null);
  const selectedName = params.get("r") ?? "";

  useEffect(() => {
    getResults()
      .then((r) => {
        const sorted = [...r.results].sort((a, b) => {
          const ka = KIND_ORDER.indexOf(a.kind);
          const kb = KIND_ORDER.indexOf(b.kind);
          return (ka === -1 ? 99 : ka) - (kb === -1 ? 99 : kb) || a.name.localeCompare(b.name);
        });
        setList(sorted);
        if (!params.get("r") && sorted[0]) setParams({ r: sorted[0].name }, { replace: true });
      })
      .catch((e: ApiError) => {
        toast.error(e.message);
        setList([]);
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const selected = list?.find((r) => r.name === selectedName) ?? null;

  useEffect(() => {
    setFull(null);
    if (!selected || selected.kind !== "probe") return;
    let alive = true;
    getResult<ProbeFull>(selected.name)
      .then((d) => {
        if (alive) setFull(d);
      })
      .catch((e: ApiError) => toast.error(`full result: ${e.message}`));
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected?.name]);

  if (list === null) {
    return (
      <div>
        <PageHeader title="Results explorer" />
        <div className="grid grid-cols-1 md:grid-cols-[260px_1fr] gap-5 items-start">
          <div className="panel p-3 flex flex-col gap-2">
            {Array.from({ length: 7 }, (_, i) => (
              <div key={i} className="skel h-9" style={{ width: `${70 + ((i * 23) % 30)}%` }} />
            ))}
          </div>
          <div className="flex flex-col gap-4">
            <div className="skel h-44" />
            <div className="skel h-24" />
            <div className="skel h-64" />
          </div>
        </div>
      </div>
    );
  }

  if (list.length === 0) {
    return (
      <div>
        <PageHeader title="Results" subtitle="Nothing in the results directory yet. Each experiment writes one JSON file that shows up here." />
        <Empty title="no results yet">
          <div className="mt-3 grid grid-cols-1 sm:grid-cols-2 gap-3 text-left">
            {KIND_ORDER.map((k) => (
              <div key={k} className="rounded-lg border border-line bg-ink/60 p-3">
                <div className="flex items-center gap-2 mb-1">
                  <span className={`chip ${KIND_TONE[k]}`}>{k}</span>
                  <span className="mono text-[12px] text-text">{KIND_CMD[k]}</span>
                </div>
                <div className="text-[12px] text-muted leading-5">{KIND_BLURB[k]}</div>
              </div>
            ))}
          </div>
        </Empty>
      </div>
    );
  }

  const byKind = KIND_ORDER.map((k) => ({ kind: k, items: list.filter((r) => r.kind === k) })).filter(
    (g) => g.items.length > 0,
  );
  const other = list.filter((r) => !KIND_ORDER.includes(r.kind));

  return (
    <div>
      <PageHeader
        title="Results explorer"
        subtitle={`${list.length} result file${list.length === 1 ? "" : "s"}. Pick one on the left; probe results load the full per-layer sweep.`}
      />
      <div className="grid grid-cols-1 md:grid-cols-[260px_1fr] gap-5 items-start">
        <nav className="panel p-2 md:sticky md:top-[72px] md:max-h-[calc(100vh-120px)] md:overflow-y-auto" aria-label="results">
          {[...byKind, ...(other.length ? [{ kind: "other", items: other }] : [])].map((g) => (
            <div key={g.kind} className="mb-2 last:mb-0">
              <div className="label px-2 pt-2 pb-1">{g.kind}</div>
              {g.items.map((r) => {
                const active = r.name === selectedName;
                return (
                  <button
                    key={r.name}
                    type="button"
                    onClick={() => setParams({ r: r.name })}
                    className={`relative w-full text-left rounded-lg px-2 py-1.5 transition ${
                      active ? "bg-panel-2 text-text" : "text-muted hover:text-text hover:bg-panel-2/60"
                    }`}
                  >
                    {active && <span className="absolute left-0 top-1.5 bottom-1.5 w-[2px] rounded bg-safe" />}
                    <div className="mono text-[12px] truncate" title={r.name}>
                      {r.name}
                    </div>
                    <div className="text-[10.5px] text-dim truncate">
                      {r.kind === "probe" && r.dataset ? `${r.dataset.replace(/^items_?/, "") || "items"} build` : (r.model ?? "no model")}
                    </div>
                  </button>
                );
              })}
            </div>
          ))}
        </nav>

        <div className="min-w-0">
          {selected ? (
            <>
              <div className="flex flex-wrap items-center gap-2 mb-4">
                <span className={`chip ${KIND_TONE[selected.kind] ?? KIND_TONE.loop}`}>{selected.kind}</span>
                <span className="mono text-[14px] text-text">{selected.name}</span>
                <span className="text-[11px] text-dim">{KIND_BLURB[selected.kind] ?? ""}</span>
              </div>
              <ResultBody summary={selected} full={full} />
            </>
          ) : (
            <Empty title="select a result" />
          )}
        </div>
      </div>
    </div>
  );
}

function ResultBody({ summary, full }: { summary: ResultSummary; full: ProbeFull | null }) {
  switch (summary.kind) {
    case "probe":
      return <ProbeView summary={summary} full={full} />;
    case "uq":
      return <UqView summary={summary} />;
    case "behavior":
      return <BehaviorView summary={summary} />;
    case "loop":
      return <LoopView summary={summary} />;
    default:
      return (
        <Empty title={`unknown result kind: ${summary.kind}`}>
          <pre className="mono text-left text-[11px] text-muted overflow-x-auto">{JSON.stringify(summary, null, 2)}</pre>
        </Empty>
      );
  }
}
