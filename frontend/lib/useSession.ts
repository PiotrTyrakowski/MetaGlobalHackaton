"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { API, ArtifactKind, Ev, api } from "./api";

/** One backend session: replays and follows its event stream. */
export function useSession() {
  const [sid, setSid] = useState<string | null>(null);
  const [plant, setPlant] = useState("Riverside Ready-Mix");
  const [events, setEvents] = useState<Ev[]>([]);
  const [error, setError] = useState<string | null>(null);
  const started = useRef(false);

  useEffect(() => {
    if (started.current) return;
    started.current = true;
    (async () => {
      // retry quietly: the server may still be starting
      for (let attempt = 0; ; attempt++) {
        try {
          let id = sessionStorage.getItem("mixpilot-sid");
          if (id && !(await api.state(id))) id = null;
          if (!id) {
            const s = await api.newSession();
            id = s.id;
            setPlant(s.plant);
            sessionStorage.setItem("mixpilot-sid", id);
          }
          setError(null);
          setSid(id);
          return;
        } catch {
          if (attempt >= 2) setError(`Can't reach the MixPilot server at ${API}. Retrying…`);
          await new Promise((r) => setTimeout(r, 1500));
        }
      }
    })();
  }, []);

  useEffect(() => {
    if (!sid) return;
    const es = new EventSource(`${API}/sessions/${sid}/events`);
    const seen = new Set<number>();
    const on = (e: MessageEvent) => {
      const ev = JSON.parse(e.data) as Ev;
      if (seen.has(ev.seq)) return;
      seen.add(ev.seq);
      setEvents((prev) => [...prev, ev]);
    };
    for (const t of ["message", "tool_start", "progress", "tool_end", "artifact", "error", "done"]) {
      es.addEventListener(t, on as EventListener);
    }
    return () => es.close();
  }, [sid]);

  const artifacts = useMemo(() => {
    const a: Partial<Record<ArtifactKind, Ev["data"]>> = {};
    for (const e of events) if (e.type === "artifact" && e.kind) a[e.kind] = e.data;
    return a;
  }, [events]);

  const busy = useMemo(() => {
    let b = false;
    for (const e of events) {
      if (e.type === "message" && e.role === "user") b = true;
      if (e.type === "done") b = false;
    }
    return b;
  }, [events]);

  const reset = async () => {
    sessionStorage.removeItem("mixpilot-sid");
    const s = await api.newSession();
    sessionStorage.setItem("mixpilot-sid", s.id);
    setEvents([]);
    setSid(s.id);
  };

  return { sid, plant, events, artifacts, busy, error, setError, reset };
}
