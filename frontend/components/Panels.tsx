"use client";

import { fmt, pct } from "@/lib/api";
import { DataArtifact, Results, Suggestion, Training } from "@/lib/types";

// ---------------------------------------------------------------- Data

export function DataPanel({ d }: { d: DataArtifact }) {
  const v = d.validation;
  const s = d.suggestions;
  const added = (s?.columns ?? []).filter((c) => c.verdict === "use");
  const collect = (s?.columns ?? []).filter((c) => c.verdict === "collect");
  const others = (s?.columns ?? []).filter((c) => ["proxy", "reject", "not_tested"].includes(c.verdict));
  const n = v.summary.batches;
  const meta = v.boxcrete_inputs.filter((i) => i.source);

  return (
    <>
      <h2>Model inputs</h2>
      <p className="lead">
        {fmt(v.summary.recipes)} recipes, {fmt(n)} batches, {fmt(v.summary.cylinders)} cylinders.
      </p>
      <div className="card" style={{ padding: "4px 10px" }}>
        <table>
          <tbody>
            <tr>
              <th>Input</th>
              <th style={{ width: "34%" }}>Coverage</th>
              <th className="num"></th>
            </tr>
            <tr>
              <td>
                Meta&apos;s inputs <span className="muted">({meta.map((i) => i.input.toLowerCase()).join(", ")})</span>
              </td>
              <td><Bar frac={1} /></td>
              <td className="num">{n}/{n}</td>
            </tr>
            {added.map((c) => (
              <tr key={c.column} className="hl-new">
                <td>{c.column} <span className="tag new">added</span></td>
                <td><Bar frac={c.coverage} /></td>
                <td className="num">{Math.round(c.coverage * n)}/{n}</td>
              </tr>
            ))}
            {collect.map((c) => (
              <tr key={c.column} className="hl-miss">
                <td>{c.column} <span className="tag miss">collect more</span></td>
                <td><Bar frac={c.coverage} warn /></td>
                <td className="num">{Math.round(c.coverage * n)}/{n}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {!s && <p className="note">Testing the other columns…</p>}

      <details className="more">
        <summary>Details</summary>
        <h3>Where each input comes from</h3>
        <div className="card" style={{ padding: "4px 10px" }}>
          <table>
            <tbody>
              {v.boxcrete_inputs.map((i) => (
                <tr key={i.input}>
                  <td>{i.input}</td>
                  <td className="muted">{i.source ?? "assumed"}</td>
                </tr>
              ))}
              {[...added, ...collect].map((c) => (
                <tr key={c.column}>
                  <td>{c.column}</td>
                  <td className="muted">{fileOf(d, c.column)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {s && others.length > 0 && (
          <>
            <h3>Checked and not used</h3>
            <div className="card" style={{ padding: "4px 10px" }}>
              <table>
                <tbody>
                  {others.map((c) => (
                    <tr key={c.column}>
                      <td>{c.column}</td>
                      <td className="muted">{whyNot(c)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="note">
              Batch columns: {s.method.batch_test}, on the scatter between batches of the same recipe. Recipe columns:{" "}
              {s.method.recipe_test}.
            </p>
          </>
        )}
        {v.issues.length > 0 && (
          <>
            <h3>Notes on the data</h3>
            <ul className="issues">
              {v.issues.map((i, k) => (
                <li key={k} className={i.level}>{i.message}</li>
              ))}
            </ul>
          </>
        )}
      </details>
    </>
  );
}

function fileOf(d: DataArtifact, col: string) {
  return d.validation.candidate_columns.find((c) => c.column === col)?.file ?? "";
}

function whyNot(c: Suggestion) {
  if (c.verdict === "proxy") {
    const a = c.associated_with?.map((x) => x.column).join(", ");
    return `Matters (p ${c.p! < 0.001 ? "< 0.001" : "= " + c.p}), but it's a label${a ? ` standing for ${a}` : ""}`;
  }
  if (c.verdict === "not_tested") return c.reason ?? "not tested";
  if (c.p !== undefined) return `No clear effect within recipes (p = ${c.p.toFixed(2)})`;
  if (c.error_with_psi !== undefined)
    return `28-day error ${fmt(c.error_without_psi)} → ${fmt(c.error_with_psi)} psi; not better than a shuffled copy`;
  return "no effect";
}

function Bar({ frac, warn }: { frac: number; warn?: boolean }) {
  return (
    <div className="bar">
      <span style={{ width: `${Math.round(frac * 100)}%`, ...(warn ? { background: "var(--warn)" } : {}) }} />
    </div>
  );
}

// ---------------------------------------------------------------- Training

export function TrainingPanel({ t }: { t: Training }) {
  const rows = t.leaderboard;
  const best = rows.find((r) => r.best)!;
  const base = rows.find((r) => r.baseline)!;
  const max = Math.max(...rows.map((r) => r.error_all_ages_pct)) * 1.08;
  const tuned = rows.filter((r) => r.tuned);
  const ages = Object.keys(base.errors_by_age_psi);

  return (
    <>
      <h2>Training on your data</h2>
      <p className="lead">Each setup is scored on recipes it never saw. Lower is better.</p>
      <div className="big">
        <div>
          <div className="n">
            {fmt(best.error_28d_psi)} <span style={{ fontSize: 14, fontWeight: 400 }}>psi</span>
          </div>
          <div className="l">28-day error</div>
        </div>
        <div>
          <div className="n" style={{ color: "var(--accent)" }}>{pct(best.vs_baseline_28d)}</div>
          <div className="l">vs Meta&apos;s default setup</div>
        </div>
      </div>
      <div className="card" style={{ padding: "4px 10px" }}>
        <table>
          <tbody>
            <tr>
              <th>Setup</th>
              <th className="num">28-day error</th>
              <th className="num">All ages</th>
              <th style={{ width: "28%" }}></th>
            </tr>
            {rows.map((r) => (
              <tr key={r.name} className={r.best ? "win" : ""}>
                <td>
                  {r.name} {r.baseline && <span className="tag base">baseline</span>}
                  {r.best && <span className="tag new">best</span>}
                </td>
                <td className="num">{fmt(r.error_28d_psi)} psi</td>
                <td className="num">{fmt(r.error_all_ages_pct, 1)}%</td>
                <td>
                  <div className="bar">
                    <span style={{ width: `${(r.error_all_ages_pct / max) * 100}%`, ...(r.best ? {} : { opacity: 0.4 }) }} />
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="note">
        {t.tuning_helped
          ? "Best is picked on all test ages. Re-tuning Meta's settings helped a little, mostly at 3 and 7 days."
          : "Best is picked on all test ages. Re-tuning Meta's settings didn't help reliably, so they're kept."}
      </p>

      <details className="more">
        <summary>Details</summary>
        <h3>By test age</h3>
        <div className="card" style={{ padding: "4px 10px" }}>
          <table>
            <tbody>
              <tr>
                <th>Setup</th>
                {ages.map((a) => (
                  <th key={a} className="num">{a.replace("d", "-day")}</th>
                ))}
                <th className="num">Folds won</th>
              </tr>
              {rows.map((r) => (
                <tr key={r.name}>
                  <td>{r.name}</td>
                  {ages.map((a) => (
                    <td key={a} className="num">{fmt(r.errors_by_age_psi[a])}</td>
                  ))}
                  <td className="num">{r.baseline ? "–" : r.folds_won_vs_baseline}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <h3>Re-tuning Meta&apos;s hard-coded settings</h3>
        <div className="card">
          <Search t={t} />
          <table style={{ marginTop: 8 }}>
            <tbody>
              <tr>
                <th>Setting</th>
                <th className="num">Meta</th>
                {tuned.map((r) => (
                  <th key={r.name} className="num">{r.inputs.length ? "Tuned, with new inputs" : "Tuned"}</th>
                ))}
              </tr>
              {(["gate_tau", "shrinkage_sigma", "mll_weight"] as const).map((k) => (
                <tr key={k}>
                  <td>{SETTING[k]}</td>
                  <td className="num">{t.meta_default[k]}</td>
                  {tuned.map((r) => (
                    <td key={r.name} className="num">{r.settings[k]}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          <p className="note">
            BoTorch, {t.trials.length} trials, {t.seconds} s. Tuned settings only count if they beat Meta&apos;s on most
            held-out folds.
          </p>
        </div>
      </details>
    </>
  );
}

const SETTING = {
  gate_tau: "Time-gate timescale τ",
  shrinkage_sigma: "Lengthscale shrinkage σ",
  mll_weight: "Training loss mix",
};

/** Best-so-far error per trial, one line per input set. */
function Search({ t }: { t: Training }) {
  const setups = [...new Set(t.trials.map((x) => x.setup))];
  const series = setups.map((s) => {
    let best = Infinity;
    return t.trials
      .filter((x) => x.setup === s)
      .map((x) => (best = Math.min(best, x.error_all_ages ?? Infinity)));
  });
  const all = series.flat().filter(Number.isFinite);
  const lo = Math.min(...all) * 0.95, hi = Math.max(...all) * 1.03;
  const W = 600, H = 150, m = { l: 44, r: 150, t: 10, b: 24 };
  const n = Math.max(...series.map((s) => s.length));
  const X = (i: number) => m.l + (i / Math.max(n - 1, 1)) * (W - m.l - m.r);
  const Y = (v: number) => H - m.b - ((v - lo) / (hi - lo)) * (H - m.t - m.b);
  const colors = ["var(--ink-3)", "var(--accent)"];
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="Best error found per trial">
      {[lo, (lo + hi) / 2, hi].map((v) => (
        <g key={v}>
          <line x1={m.l} x2={W - m.r} y1={Y(v)} y2={Y(v)} stroke="var(--grid)" />
          <text x={m.l - 6} y={Y(v) + 4} textAnchor="end" fontSize="10.5" fill="var(--ink-3)">{(v * 100).toFixed(1)}%</text>
        </g>
      ))}
      <text x={m.l} y={H - 6} fontSize="10.5" fill="var(--ink-3)">trial 1 (Meta&apos;s settings)</text>
      <text x={W - m.r} y={H - 6} textAnchor="end" fontSize="10.5" fill="var(--ink-3)">trial {n}</text>
      {series.map((s, k) => (
        <g key={k}>
          <polyline fill="none" stroke={colors[k % 2]} strokeWidth="2" points={s.map((v, i) => `${X(i)},${Y(v)}`).join(" ")} />
          <text x={X(s.length - 1) + 8} y={Y(s[s.length - 1]) + 4} fontSize="11" fill="var(--ink-2)">{setups[k]}</text>
        </g>
      ))}
    </svg>
  );
}

// ---------------------------------------------------------------- Results

export function ResultsPanel({ r }: { r: Results }) {
  const pts = r.recipes.filter((x) => x.observed_28d_psi !== null);
  const xs = [...pts.map((p) => p.gwp_kg_m3), ...r.proposals.map((p) => p.gwp_kg_m3)];
  const ys = [...pts.map((p) => p.observed_28d_psi!), ...r.proposals.map((p) => p.predicted_28d_psi)];
  const [x0, x1] = niceRange(Math.min(...xs), Math.max(...xs), 50);
  const [y0, y1] = niceRange(Math.min(...ys), Math.max(...ys), 1000);
  const W = 640, H = 340, m = { l: 56, r: 24, t: 30, b: 40 };
  const X = (v: number) => m.l + ((v - x0) / (x1 - x0)) * (W - m.l - m.r);
  const Y = (v: number) => H - m.b - ((v - y0) / (y1 - y0)) * (H - m.t - m.b);
  const byId = new Map(r.recipes.map((x) => [x.recipe, x]));
  const avgSaving = r.proposals.length ? r.proposals.reduce((a, p) => a + p.co2_saving, 0) / r.proposals.length : 0;
  const unit = r.proposals[0]?.composition_unit === "lb/yd3" ? "lb/yd³" : "kg/m³";
  const extras = Object.keys(r.proposals[0]?.extras ?? {});

  return (
    <>
      <h2>Lower-CO₂ mixes</h2>
      <p className="lead">
        Mixes to test next: the same strength as your recipe{r.proposals.length ? <>, about <b>{fmt(avgSaving * 100)}% less CO₂</b></> : null}.
      </p>
      <div className="card">
        <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="CO2 versus 28-day strength">
          {ticks(y0, y1, 1000).map((t) => (
            <g key={t}>
              <line x1={m.l} x2={W - m.r} y1={Y(t)} y2={Y(t)} stroke="var(--grid)" />
              <text x={m.l - 8} y={Y(t) + 4} textAnchor="end" fontSize="11" fill="var(--ink-3)">{fmt(t)}</text>
            </g>
          ))}
          {ticks(x0, x1, 50).map((t) => (
            <text key={t} x={X(t)} y={H - m.b + 16} textAnchor="middle" fontSize="11" fill="var(--ink-3)">{t}</text>
          ))}
          <text x={(m.l + W - m.r) / 2} y={H - 4} textAnchor="middle" fontSize="11.5" fill="var(--ink-2)">
            CO₂ (kg/m³) · lower is better
          </text>
          <text x={12} y={m.t - 12} fontSize="11.5" fill="var(--ink-2)">28-day strength (psi)</text>
          {pts.map((p) => (
            <circle key={p.recipe} cx={X(p.gwp_kg_m3)} cy={Y(p.observed_28d_psi!)} r="4" fill="var(--ink-3)" opacity=".45">
              <title>{`${p.recipe}: ${fmt(p.observed_28d_psi)} psi, ${p.gwp_kg_m3} kg CO₂/m³`}</title>
            </circle>
          ))}
          {r.proposals.map((p, i) => {
            const from = byId.get(p.replaces);
            const dy = [-10, 18, -10][i % 3], dx = [-8, 8, 10][i % 3];
            return (
              <g key={p.id}>
                {from && from.observed_28d_psi !== null && (
                  <line x1={X(from.gwp_kg_m3)} y1={Y(from.observed_28d_psi)} x2={X(p.gwp_kg_m3)} y2={Y(p.predicted_28d_psi)}
                        stroke="var(--accent)" strokeDasharray="3 3" opacity=".6" />
                )}
                <circle cx={X(p.gwp_kg_m3)} cy={Y(p.predicted_28d_psi)} r="6" fill="var(--accent)">
                  <title>{`${p.id}: ${fmt(p.predicted_28d_psi)} psi, ${p.gwp_kg_m3} kg CO₂/m³`}</title>
                </circle>
                <text x={X(p.gwp_kg_m3) + dx} y={Y(p.predicted_28d_psi) + dy} textAnchor={dx < 0 ? "end" : "start"}
                      fontSize="12" fontWeight="500" fill="var(--ink)">{p.id}</text>
              </g>
            );
          })}
        </svg>
        <div className="legend">
          <span><i style={{ background: "var(--ink-3)", opacity: 0.45 }} />your recipes</span>
          <span><i style={{ background: "var(--accent)" }} />proposed</span>
        </div>
      </div>
      {r.proposals.length > 0 ? (
        <div className="card" style={{ padding: "4px 10px" }}>
          <table>
            <tbody>
              <tr>
                <th>Mix</th>
                <th>Replaces</th>
                <th className="num">Predicted 28-day</th>
                <th className="num">CO₂</th>
              </tr>
              {r.proposals.map((p) => (
                <tr key={p.id}>
                  <td>{p.id}</td>
                  <td className="mono" style={{ fontSize: 12.5 }}>{p.replaces}</td>
                  <td className="num">{fmt(p.predicted_28d_psi)} psi</td>
                  <td className="num" style={{ color: "var(--accent)" }}>{pct(-p.co2_saving)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="note">No mix cleared the bar this time.</p>
      )}
      {r.proposals.length > 0 && (
        <details className="more">
          <summary>Details</summary>
          <h3>Recipes ({unit})</h3>
          <div className="card" style={{ padding: "4px 10px", overflowX: "auto" }}>
            <table>
              <tbody>
                <tr>
                  <th>Mix</th>
                  {Object.keys(r.proposals[0].composition).map((k) => (
                    <th key={k} className="num">{k}</th>
                  ))}
                  {extras.map((e) => (
                    <th key={e} className="num">{e.replace(/ \(.*\)/, "")}</th>
                  ))}
                  <th className="num">± (90%)</th>
                  <th className="num">kg CO₂/m³</th>
                </tr>
                {r.proposals.map((p) => (
                  <tr key={p.id}>
                    <td>{p.id}</td>
                    {Object.values(p.composition).map((v, i) => (
                      <td key={i} className="num">{fmt(v)}</td>
                    ))}
                    {extras.map((e) => (
                      <td key={e} className="num">{fmt(p.extras[e], 1)}</td>
                    ))}
                    <td className="num">{fmt(1.645 * p.sd_28d_psi)}</td>
                    <td className="num">{p.gwp_kg_m3} <span className="muted">vs {p.recipe_gwp_kg_m3}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="note">
            A mix counts only if the model&apos;s cautious estimate (90% lower bound) still reaches the recipe&apos;s
            strength. Air is kept at the recipe&apos;s value. CO₂: {r.method.co2}.
          </p>
        </details>
      )}
      <p className="note">Lab-cured predictions. Your QC engineer signs off before any trial batch.</p>
    </>
  );
}

function niceRange(lo: number, hi: number, step: number): [number, number] {
  return [Math.floor(lo / step) * step, Math.ceil(hi / step) * step];
}

function ticks(lo: number, hi: number, step: number) {
  const out = [];
  for (let v = lo; v <= hi + 1e-9; v += step) out.push(Math.round(v));
  return out;
}

