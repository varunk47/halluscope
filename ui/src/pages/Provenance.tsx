import { useEffect, useState, type ReactNode } from "react";
import { ApiError, getProvenance, type Provenance as Prov } from "../api";
import { Empty, PageHeader, Panel, SectionLabel, Spinner } from "../components/Primitives";
import { useToast } from "../components/Toast";

export default function Provenance() {
  const toast = useToast();
  const [data, setData] = useState<Prov | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    getProvenance()
      .then(setData)
      .catch((e: ApiError) => {
        toast.error(e.message);
        setFailed(true);
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (failed) {
    return (
      <div>
        <PageHeader title="Provenance" />
        <Empty title="could not load provenance">Is the backend running on port 8000?</Empty>
      </div>
    );
  }
  if (!data) {
    return (
      <div>
        <PageHeader title="Provenance" />
        <Spinner label="loading" />
      </div>
    );
  }

  const aliases = Object.entries(data.judge_aliases ?? {});
  const calls = Object.entries(data.llm_calls ?? {});

  return (
    <div>
      <PageHeader
        title="Provenance"
        subtitle="Everything needed to reproduce a run: library versions, hardware, seeds, split fractions, and what the judge models cost."
      />

      <div className="grid grid-cols-1 md:grid-cols-2 gap-5 items-start">
        <div className="flex flex-col gap-5">
          <Panel>
            <SectionLabel>environment</SectionLabel>
            <Sheet
              rows={[
                ["python", data.python],
                ["torch", data.torch],
                ["transformers", data.transformers],
                ["platform", data.platform],
                [
                  "cuda",
                  <span className={data.cuda ? "text-safe" : "text-risk"}>{data.cuda ? "available" : "unavailable"}</span>,
                ],
                ["gpu", data.gpu ?? <span className="text-dim">none</span>],
              ]}
            />
          </Panel>

          <Panel>
            <SectionLabel>experiment settings</SectionLabel>
            <Sheet
              rows={[
                ["seed", data.seed],
                [
                  "split fractions",
                  <span>
                    {data.split_fractions.map((f, i) => (
                      <span key={i}>
                        {i > 0 && <span className="text-dim"> / </span>}
                        {f}
                      </span>
                    ))}
                    <span className="ml-2 text-[10px] text-dim font-sans">train / val / test</span>
                  </span>,
                ],
                ["n bootstrap", data.n_bootstrap],
                ["gate alpha", data.gate_alpha],
              ]}
            />
          </Panel>
        </div>

        <div className="flex flex-col gap-5">
          <Panel padded={false}>
            <div className="px-5 pt-4">
              <SectionLabel right={`${aliases.length} alias${aliases.length === 1 ? "" : "es"}`}>judge aliases</SectionLabel>
            </div>
            {aliases.length === 0 ? (
              <div className="px-5 pb-4 text-[12px] text-dim">no judge aliases configured</div>
            ) : (
              <table className="w-full text-[12.5px]">
                <thead>
                  <tr className="label border-b border-line">
                    <th className="text-left font-medium px-5 py-2">alias</th>
                    <th className="text-left font-medium px-3 py-2 pr-5">models, in fallback order</th>
                  </tr>
                </thead>
                <tbody>
                  {aliases.map(([alias, models]) => (
                    <tr key={alias} className="border-b border-line/60 last:border-b-0 align-top">
                      <td className="px-5 py-2 mono text-text whitespace-nowrap">{alias}</td>
                      <td className="px-3 py-2 pr-5">
                        <div className="flex flex-wrap gap-1.5">
                          {models.map((m, i) => (
                            <span key={m} className="chip border-line text-muted mono">
                              <span className="text-dim">{i + 1}</span>
                              {m}
                            </span>
                          ))}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </Panel>

          <Panel padded={false}>
            <div className="px-5 pt-4">
              <SectionLabel
                right={
                  <span>
                    total <span className="mono text-text">${data.total_cost_usd.toFixed(4)}</span>
                  </span>
                }
              >
                llm calls
              </SectionLabel>
            </div>
            {calls.length === 0 ? (
              <div className="px-5 pb-4 text-[12px] text-dim">no calls logged yet</div>
            ) : (
              <table className="w-full text-[12.5px]">
                <thead>
                  <tr className="label border-b border-line">
                    <th className="text-left font-medium px-5 py-2">alias</th>
                    <th className="text-right font-medium px-3 py-2">calls</th>
                    <th className="text-right font-medium px-3 py-2">ok</th>
                    <th className="text-right font-medium px-3 py-2">cost usd</th>
                    <th className="text-left font-medium px-3 py-2 pr-5">answered (ok of calls, usd)</th>
                  </tr>
                </thead>
                <tbody className="mono">
                  {calls.map(([alias, c]) => (
                    <tr key={alias} className="border-b border-line/60 last:border-b-0 align-top">
                      <td className="px-5 py-2 text-text">{alias}</td>
                      <td className="text-right px-3 py-2">{c.calls}</td>
                      <td className={`text-right px-3 py-2 ${c.ok === c.calls ? "text-safe" : "text-risk"}`}>{c.ok}</td>
                      <td className="text-right px-3 py-2">{c.cost_usd.toFixed(4)}</td>
                      <td className="px-3 py-2 pr-5 text-[11px]">
                        <div className="flex flex-col gap-0.5">
                          {(c.answered ?? c.models).map((m) => {
                            const pm = c.per_model?.[m];
                            return (
                              <div key={m} className="flex gap-2">
                                <span className="text-text">{m}</span>
                                {pm && (
                                  <span className="text-dim">
                                    {pm.ok} of {pm.calls}, {pm.cost_usd.toFixed(2)}
                                  </span>
                                )}
                              </div>
                            );
                          })}
                          {c.refused && c.refused.length > 0 && (
                            <div className="text-dim mt-1">
                              refused every call: {c.refused.join(", ")}
                            </div>
                          )}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </Panel>
        </div>
      </div>
    </div>
  );
}

function Sheet({ rows }: { rows: [string, ReactNode][] }) {
  return (
    <dl className="grid grid-cols-[minmax(120px,auto)_1fr] gap-x-6">
      {rows.map(([k, v]) => (
        <div key={k} className="contents">
          <dt className="text-[12px] text-muted py-2 border-b border-line/60">{k}</dt>
          <dd className="mono text-[13px] text-text py-2 border-b border-line/60 break-all">{v}</dd>
        </div>
      ))}
    </dl>
  );
}
