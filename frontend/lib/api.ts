export const API = process.env.NEXT_PUBLIC_API ?? "http://localhost:8811";

export type Ev = {
  seq: number;
  type: "text_delta" | "message" | "tool_start" | "progress" | "tool_end" | "artifact" | "error" | "done";
  role?: "user" | "assistant";
  text?: string;
  phase?: string;
  stream?: number;
  tool?: string;
  label?: string;
  seconds?: number;
  ok?: boolean;
  kind?: ArtifactKind;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  data?: any;
};

export type ArtifactKind = "data" | "training" | "results" | "models";

async function post(path: string, body?: unknown) {
  const r = await fetch(`${API}${path}`, {
    method: "POST",
    headers: body === undefined ? {} : { "content-type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${path}: ${r.status} ${await r.text()}`);
  return r.json();
}

export const api = {
  newSession: (plant?: string) => post("/sessions", plant ? { plant } : {}) as Promise<{ id: string; plant: string }>,
  state: async (sid: string) => {
    const r = await fetch(`${API}/sessions/${sid}/state`);
    return r.ok ? r.json() : null;
  },
  upload: async (sid: string, files: File[]) => {
    const fd = new FormData();
    files.forEach((f) => fd.append("files", f));
    const r = await fetch(`${API}/sessions/${sid}/files`, { method: "POST", body: fd });
    if (!r.ok) throw new Error(await r.text());
    return (await r.json()) as { files: string[] };
  },
  demoFiles: (sid: string) => post(`/sessions/${sid}/demo-files`) as Promise<{ files: string[] }>,
  message: (sid: string, text: string) => post(`/sessions/${sid}/messages`, { text }),
  approve: (sid: string, inputs: string[]) => post(`/sessions/${sid}/approve`, { inputs }),
  models: async (plant: string) => (await fetch(`${API}/models?plant=${encodeURIComponent(plant)}`)).json(),
};

export const fmt = (n: number | null | undefined, digits = 0) =>
  n === null || n === undefined ? "–" : n.toLocaleString("en-US", { maximumFractionDigits: digits, minimumFractionDigits: digits });

export const pct = (x: number, digits = 0) => `${x > 0 ? "+" : x < 0 ? "−" : ""}${Math.abs(x * 100).toFixed(digits)}%`;
