"use client";

import { useEffect, useState } from "react";
import { ArtifactKind, api, fmt, pct } from "@/lib/api";
import { DataArtifact, ModelMeta, Results, Training } from "@/lib/types";
import { useSession } from "@/lib/useSession";
import Chat from "./Chat";
import { DataPanel, ResultsPanel, TrainingPanel } from "./Panels";

const TABS: [ArtifactKind, string][] = [
  ["data", "Data"],
  ["training", "Training"],
  ["results", "Results"],
];

export default function App() {
  const { sid, plant, events, artifacts, busy, error, setError, reset } = useSession();
  const [view, setView] = useState<"project" | "models">("project");
  const [picked, setPicked] = useState<{ k: ArtifactKind; seq: number } | null>(null);
  const [composer, setComposer] = useState("");
  const [approvedAt, setApprovedAt] = useState<number | null>(null);

  // a panel opens when it arrives; the user's own pick wins until the next one arrives
  const latest = [...events].reverse().find((e) => e.type === "artifact" && e.kind !== "models");
  const lastSeq = events.length ? events[events.length - 1].seq : -1;
  const tab = picked && picked.seq >= (latest?.seq ?? -1) ? picked.k : latest?.kind ?? null;
  const setTab = (k: ArtifactKind) => setPicked({ k, seq: lastSeq });

  const lastSuggestSeq = [...events].reverse().find((e) => e.type === "artifact" && e.kind === "data" && e.data?.suggestions)?.seq;
  const approved = approvedAt !== null && lastSuggestSeq !== undefined && approvedAt > lastSuggestSeq;

  const guard = async (f: () => Promise<unknown>) => {
    try {
      setError(null);
      await f();
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e));
    }
  };

  const sendFiles = (names: string[]) =>
    api.message(sid!, `Here are our lab files: ${names.join(", ")}`);

  return (
    <div className="app">
      <header className="top">
        <div className="logo"><i />MixPilot</div>
        <div className="proj">{plant}</div>
        <nav className="nav">
          <button className={view === "project" ? "on" : ""} onClick={() => setView("project")}>Project</button>
          <button className={view === "models" ? "on" : ""} onClick={() => setView("models")}>Models</button>
          <button title="Start over with a new session" onClick={() => guard(async () => { await reset(); setPicked(null); setApprovedAt(null); })}>
            New
          </button>
        </nav>
      </header>

      {view === "project" ? (
        <main className="view on" id="project">
          <Chat
            events={events}
            busy={busy || !sid}
            activeTab={tab}
            approved={approved}
            onOpen={setTab}
            onSend={(t) => guard(() => api.message(sid!, t))}
            onApprove={(inputs) =>
              guard(async () => {
                setApprovedAt(events.length ? events[events.length - 1].seq : 0);
                await api.approve(sid!, inputs);
              })
            }
            onFiles={(files) =>
              guard(async () => {
                if (!files.length) return;
                const r = await api.upload(sid!, files);
                await sendFiles(r.files);
              })
            }
            onDemo={(set) =>
              guard(async () => {
                const r = await api.demoFiles(sid!, set);
                await sendFiles(r.files);
              })
            }
            composerText={composer}
            setComposerText={setComposer}
            error={error}
          />
          <section className="pane">
            <nav className="ptabs">
              {TABS.filter(([k]) => artifacts[k]).map(([k, label]) => (
                <button key={k} className={`ptab ${tab === k ? "on" : ""}`} onClick={() => setTab(k)}>{label}</button>
              ))}
            </nav>
            <div className="pbody">
              {tab === "data" && artifacts.data && <DataPanel d={artifacts.data as DataArtifact} />}
              {tab === "training" && artifacts.training && <TrainingPanel t={artifacts.training as Training} />}
              {tab === "results" && artifacts.results && <ResultsPanel r={artifacts.results as Results} />}
              {!tab && (
                <div className="empty">
                  The Data, Training and Results panels appear here as MixPilot works through your files.
                </div>
              )}
            </div>
          </section>
        </main>
      ) : (
        <Models
          plant={plant}
          refreshKey={artifacts.models}
          onUse={(v) => {
            setView("project");
            setComposer(`Using model ${v}: what will `);
          }}
        />
      )}
    </div>
  );
}

function Models({ plant, refreshKey, onUse }: { plant: string; refreshKey: unknown; onUse: (v: string) => void }) {
  const [models, setModels] = useState<ModelMeta[] | null>(null);
  useEffect(() => {
    api.models(plant).then((r) => setModels(r.models)).catch(() => setModels([]));
  }, [plant, refreshKey]);
  return (
    <main className="view on" id="models">
      <div className="wrap">
        <h2 style={{ margin: "0 0 4px", fontSize: 20 }}>Models</h2>
        <p className="lead">
          Models trained on {plant}&apos;s data. Use one in chat to predict a mix; upload new results to retrain.
        </p>
        <div className="card" style={{ padding: "4px 10px" }}>
          <table>
            <tbody>
              <tr>
                <th>Version</th>
                <th>Trained</th>
                <th>Inputs</th>
                <th className="num">Recipes</th>
                <th className="num">28-day error</th>
                <th className="num">vs Meta</th>
                <th></th>
              </tr>
              {(models ?? []).slice().reverse().map((m) => (
                <tr key={m.version} className={m.active ? "win" : ""}>
                  <td>
                    {m.version} {m.active && <span className="tag new">active</span>}
                  </td>
                  <td>{new Date(m.trained).toLocaleString("en-US", { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" })}</td>
                  <td>Meta&apos;s 10{m.inputs.length ? ` + ${m.inputs.map((i) => i.replace(/ \(.*\)/, "").toLowerCase()).join(", ")}` : ""}</td>
                  <td className="num">{m.recipes}</td>
                  <td className="num">{fmt(m.error_28d_psi)} psi</td>
                  <td className="num">{pct(m.vs_meta_default)}</td>
                  <td style={{ textAlign: "right" }}>
                    <button className="btn" style={{ padding: "5px 12px" }} onClick={() => onUse(m.version)}>Use in chat</button>
                  </td>
                </tr>
              ))}
              {models && models.length === 0 && (
                <tr>
                  <td colSpan={7} className="muted">No models yet. Train one from the Project tab.</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </main>
  );
}
