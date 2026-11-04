"use client";

import { useEffect, useRef, useState } from "react";
import { ArtifactKind, Ev, fmt } from "@/lib/api";
import { Suggestion, Suggestions } from "@/lib/types";

type Step = { tool: string; label: string; done: boolean; ok: boolean; seconds?: number; progress?: string };
type Item =
  | { k: "stream"; id: number; text: string }
  | { k: "comment"; text: string }
  | { k: "step"; step: Step }
  | { k: "final"; text: string }
  | { k: "error"; text: string };
type Turn = { user?: string; items: Item[]; artifacts: ArtifactKind[]; suggestions?: Suggestions; done: boolean };

const ART: Record<ArtifactKind, string> = { data: "Data", training: "Training", results: "Results", models: "Models" };

function turns(events: Ev[]): Turn[] {
  const out: Turn[] = [];
  let cur: Turn = { items: [], artifacts: [], done: true };
  const steps: Record<string, Step> = {};
  for (const e of events) {
    if (e.type === "message" && e.role === "user") {
      if (cur.user !== undefined || cur.items.length) out.push(cur);
      cur = { user: e.text, items: [], artifacts: [], done: false };
    } else if (e.type === "text_delta") {
      const open = cur.items.find((x) => x.k === "stream" && x.id === e.stream);
      if (open?.k === "stream") open.text += e.text;
      else cur.items.push({ k: "stream", id: e.stream!, text: e.text! });
    } else if (e.type === "message") {
      const item: Item = e.phase === "commentary" ? { k: "comment", text: e.text! } : { k: "final", text: e.text! };
      const at = cur.items.findIndex((x) => x.k === "stream" && x.id === e.stream);
      if (at >= 0) cur.items[at] = item;
      else cur.items.push(item);
    } else if (e.type === "tool_start") {
      const step: Step = { tool: e.tool!, label: e.label ?? e.tool!, done: false, ok: true };
      steps[e.tool!] = step;
      cur.items.push({ k: "step", step });
    } else if (e.type === "progress" && steps[e.tool!]) {
      steps[e.tool!].progress = e.text;
    } else if (e.type === "tool_end" && steps[e.tool!]) {
      Object.assign(steps[e.tool!], { done: true, ok: e.ok !== false, seconds: e.seconds });
    } else if (e.type === "artifact" && e.kind) {
      if (e.kind !== "models" && !cur.artifacts.includes(e.kind)) cur.artifacts.push(e.kind);
      if (e.kind === "data" && e.data?.suggestions) cur.suggestions = e.data.suggestions;
    } else if (e.type === "error") {
      cur.items.push({ k: "error", text: e.text! });
    } else if (e.type === "done") {
      cur.done = true;
    }
  }
  if (cur.user !== undefined || cur.items.length) out.push(cur);
  return out;
}

export default function Chat(props: {
  events: Ev[];
  busy: boolean;
  activeTab: ArtifactKind | null;
  approved: boolean;
  onOpen: (k: ArtifactKind) => void;
  onSend: (text: string) => void;
  onApprove: (inputs: string[]) => void;
  onFiles: (files: File[]) => void;
  onDemo: () => void;
  composerText: string;
  setComposerText: (t: string) => void;
  error: string | null;
}) {
  const ts = turns(props.events);
  const end = useRef<HTMLDivElement>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  const hasFiles = ts.some((t) => t.user);
  const lastSuggest = [...ts].reverse().find((t) => t.suggestions);
  const trained = lastSuggest && ts.slice(ts.indexOf(lastSuggest)).some((t) => t.artifacts.includes("training"));
  const offerTraining = lastSuggest && !trained && !props.approved;

  useEffect(() => end.current?.scrollIntoView({ block: "end" }), [props.events.length]);

  const send = () => {
    const t = props.composerText.trim();
    if (t && !props.busy) {
      props.onSend(t);
      props.setComposerText("");
    }
  };

  return (
    <section className="chat">
      <div className="stream">
        {!hasFiles && (
          <div className="agent">
            <p>
              Send me your plant&apos;s lab files: mix designs, cylinder breaks and the QC log (Excel or CSV). I&apos;ll
              retrain Meta&apos;s BOxCrete model on them and look for columns that make it better.
            </p>
            <div
              className={`drop ${over ? "over" : ""}`}
              onDragOver={(e) => (e.preventDefault(), setOver(true))}
              onDragLeave={() => setOver(false)}
              onDrop={(e) => {
                e.preventDefault();
                setOver(false);
                if (!props.busy) props.onFiles(Array.from(e.dataTransfer.files));
              }}
            >
              <div>Drop Excel or CSV files here</div>
              <div className="row">
                <button className="btn ghost" disabled={props.busy} onClick={() => fileInput.current?.click()}>
                  Choose files
                </button>
                <button className="btn" disabled={props.busy} onClick={() => props.onDemo()}>
                  Use demo plant files
                </button>
              </div>
              <div className="muted">Demo: Riverside Ready-Mix, synthetic data</div>
            </div>
          </div>
        )}

        {ts.map((t, i) => (
          <div key={i} style={{ display: "contents" }}>
            {t.user && <UserMsg text={t.user} />}
            {(t.items.length > 0 || !t.done) && (
              <div className="agent">
                <TurnBody turn={t} />
                {t.done && t.artifacts.length > 0 && (
                  <div className="links">
                    {t.artifacts.map((k) => (
                      <button key={k} className={props.activeTab === k ? "on" : ""} onClick={() => props.onOpen(k)}>
                        {ART[k]} ›
                      </button>
                    ))}
                  </div>
                )}
                {t.suggestions && t.done && <Findings s={t.suggestions} />}
                {offerTraining && t.done && i === ts.length - 1 && (
                  <button className="btn" disabled={props.busy} onClick={() => props.onApprove(lastSuggest.suggestions!.use)}>
                    Train model
                  </button>
                )}
              </div>
            )}
          </div>
        ))}
        {props.error && <div className="err">{props.error}</div>}
        <div ref={end} />
      </div>
      <div className="composer">
        <button className="attach" title="Add files" onClick={() => fileInput.current?.click()}>
          ＋
        </button>
        <input
          placeholder={props.busy ? "MixPilot is working…" : "Ask about your mixes, or add files…"}
          value={props.composerText}
          onChange={(e) => props.setComposerText(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && send()}
        />
        <button className="btn" disabled={props.busy || !props.composerText.trim()} onClick={send}>
          Send
        </button>
        <input
          ref={fileInput}
          type="file"
          multiple
          accept=".xlsx,.xls,.csv"
          hidden
          onChange={(e) => {
            props.onFiles(Array.from(e.target.files ?? []));
            e.target.value = "";
          }}
        />
      </div>
    </section>
  );
}

function UserMsg({ text }: { text: string }) {
  const files = text.match(/[\w.\- ]+\.(xlsx|xls|csv)/gi)?.map((f) => f.trim()) ?? [];
  const body = files.length ? text.split(":")[0] + "." : text;
  return (
    <div className="user">
      {body}
      {files.length > 0 && (
        <div className="files">
          {files.map((f) => (
            <span key={f} className="file">{f}</span>
          ))}
        </div>
      )}
    </div>
  );
}

function TurnBody({ turn }: { turn: Turn }) {
  const steps = turn.items.flatMap((x) => (x.k === "step" ? [x.step] : []));
  const finals = turn.items.filter((x) => x.k === "final" || x.k === "error" || x.k === "stream");
  const comments = turn.items.filter((x) => x.k === "comment");
  const running = steps.find((x) => !x.done);
  const done = steps.filter((x) => x.done);
  const total = done.reduce((a, x) => a + (x.seconds ?? 0), 0);

  return (
    <>
      {turn.done && steps.length > 0 && (
        <details className="steps-done">
          <summary>
            ✓ {steps.length} {steps.length === 1 ? "step" : "steps"}
            {total >= 1 ? ` · ${fmt(total)} s` : ""}
          </summary>
          <div className="steps">
            {steps.map((x, i) => (
              <div key={i} className="step">
                <span className="ic">{x.ok ? "✓" : "!"}</span>
                {x.label}
                {(x.seconds ?? 0) >= 1 && <span className="sec">{fmt(x.seconds!)} s</span>}
              </div>
            ))}
          </div>
        </details>
      )}
      {!turn.done && (
        <div className="steps">
          {done.length > 0 && <div className="step"><span className="ic">✓</span>{done.map((x) => x.label).join(" · ")}</div>}
          <div className="step run">
            <span className="ic"><span className="spin" /></span>
            {running?.label ?? (comments.length ? comments[comments.length - 1].text : "Thinking")}
          </div>
          {running?.progress && <div className="step-prog">{running.progress}</div>}
        </div>
      )}
      {finals.map((x, i) =>
        x.k === "error" ? <div key={i} className="err">{x.text}</div> : <p key={i}>{x.k === "final" || x.k === "stream" ? x.text : ""}</p>,
      )}
    </>
  );
}

function Findings({ s }: { s: Suggestions }) {
  const use = s.columns.filter((c) => c.verdict === "use");
  const collect = s.columns.filter((c) => c.verdict === "collect");
  return (
    <>
      {use.map((c) => (
        <div key={c.column} className="find new">
          <div className="h">{c.column}: ready to use</div>
          <div className="d">{findingDetail(c)}</div>
        </div>
      ))}
      {collect.map((c) => (
        <div key={c.column} className="find miss">
          <div className="h">
            {c.column}: {fmt(c.coverage * 100)}% recorded
          </div>
          <div className="d">{collectDetail(c)}</div>
        </div>
      ))}
    </>
  );
}

export function findingDetail(c: Suggestion) {
  if (c.level === "batch" && c.scatter_explained !== undefined) {
    const per = c.pct_strength_per_unit;
    return `Explains ${fmt(c.scatter_explained * 100)}% of the batch-to-batch scatter${
      per !== undefined ? ` · ${per > 0 ? "+" : "−"}${fmt(Math.abs(per), 1)}% strength per unit` : ""
    }`;
  }
  if (c.error_without_psi !== undefined) {
    return `28-day error ${fmt(c.error_without_psi)} → ${fmt(c.error_with_psi)} psi`;
  }
  return "";
}

export function collectDetail(c: Suggestion) {
  const parts = [];
  if (c.missing_by) parts.push(`Missing for ${c.missing_by.missing_values.join(", ")}. Record it from now on.`);
  if (c.gain !== undefined) parts.push(`Worth −${fmt(c.gain * 100)}% error where recorded.`);
  return parts.join(" ");
}
