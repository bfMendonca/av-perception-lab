// Readiness of the edited pipeline's models: a debounced `POST
// /api/models/status` (a response to an older config is dropped: the
// sequence is bumped when a request is scheduled, as in the validation), the
// Prepare jobs this page started (SSE progress per model), and the outcome of
// the last attempt per model (a failed Prepare stays "failed" with its reason
// after the refresh, until it is retried or the model becomes ready).
import { createContext, createElement, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { StatusRejected, type ModelKey, type ModelState, type ModelStatus, type ModelsApi, type PrepareEvent, type PrepareJob } from "../api/models";
import type { ConfigDict } from "../api/types";
import { keyOf, plural, preparable, rowsForSlot, sizeOf, where } from "./readiness";

const PREPARING_POLL_MS = 2000;
const ACTIVE = new Set(["queued", "preparing"]);

export type ReadinessStatus = "idle" | "pending" | "ready" | "invalid" | "unavailable";

/** Byte progress of one model being prepared (`fraction` null when no total is known). */
export interface ModelProgress {
  done: number;
  total: number | null;
  fraction: number | null;
  /** The item being fetched (asset key, `code:<name>`). */
  item: string | null;
}

/** A status row as the UI shows it: the server's row with this page's Prepare overlaid. */
export interface ModelView extends ModelStatus {
  /** The state and message the server reported (before the overlay). */
  serverState: ModelState;
  serverMessage: string | null;
  progress: ModelProgress | null;
  /** The last Prepare of this model ended without making it ready. */
  attempt: { state: ModelState; message: string | null } | null;
}

export interface ReadinessNotice {
  tone: "info" | "warn" | "error";
  text: string;
}

export interface UseModelStatusOptions {
  api: ModelsApi;
  /** Delay of the status request after the last change (ms). */
  debounceMs?: number;
  notify?: (notice: ReadinessNotice) => void;
}

interface Local {
  state: ModelState;
  jobId: string;
  items: Record<string, { done: number; total: number | null }>;
}

interface Attempt {
  state: ModelState;
  message: string | null;
  /** Identity of the model it was about (a changed type or name drops it). */
  ident: string;
}

const identOf = (r: { type: string; name: string }) => `${r.type}\u0000${r.name}`;

function progressOf(local: Local | undefined): ModelProgress | null {
  if (!local) return null;
  const items = Object.entries(local.items);
  let done = 0;
  let total = 0;
  let known = items.length > 0;
  for (const [, p] of items) {
    done += p.done;
    if (p.total === null) known = false;
    else total += p.total;
  }
  const last = items.length ? items[items.length - 1][0] : null;
  return { done, total: known ? total : null, fraction: known && total > 0 ? Math.min(1, done / total) : null, item: last };
}

export function useModelStatus(config: ConfigDict | null, options: UseModelStatusOptions) {
  const { api, debounceMs = 400 } = options;
  const notify = useRef(options.notify);
  notify.current = options.notify;
  const configText = config === null ? null : JSON.stringify(config);
  const configRef = useRef(config);
  configRef.current = config;

  const [status, setStatus] = useState<ReadinessStatus>("idle");
  const [message, setMessage] = useState<string | null>(null);
  const [rows, setRows] = useState<ModelStatus[]>([]);
  const rowsRef = useRef(rows);
  rowsRef.current = rows;
  // A model already ready is a no-op for the server: never shown as preparing.
  const isReady = (k: string) => rowsRef.current.some((r) => keyOf(r) === k && r.state === "ready");
  const [local, setLocal] = useState<Record<string, Local>>({});
  const [attempts, setAttempts] = useState<Record<string, Attempt>>({});
  const [activeJobs, setActiveJobs] = useState<string[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [tick, setTick] = useState(0);
  const seq = useRef(0);
  const subscriptions = useRef(new Map<string, () => void>());

  const refresh = useCallback(() => setTick((n) => n + 1), []);

  // Debounced status of the current config; stale answers are dropped.
  useEffect(() => {
    seq.current++; // a response to any request already in flight is now stale
    if (configText === null) {
      setStatus("idle");
      setRows([]);
      setMessage(null);
      return;
    }
    setStatus("pending"); // the last rows stay on screen until the answer (no flicker)
    const body = JSON.parse(configText) as ConfigDict;
    const timer = setTimeout(() => {
      const mine = ++seq.current;
      api.status(body).then(
        (report) => {
          if (mine !== seq.current) return;
          setRows(report.models);
          setStatus("ready");
          setMessage(null);
        },
        (exc: unknown) => {
          if (mine !== seq.current) return;
          setRows([]);
          setStatus(exc instanceof StatusRejected ? "invalid" : "unavailable");
          setMessage(exc instanceof Error ? exc.message : String(exc));
        },
      );
    }, debounceMs);
    return () => clearTimeout(timer);
  }, [api, configText, debounceMs, tick]);

  const finishJob = useCallback(
    (id: string) => {
      subscriptions.current.get(id)?.();
      subscriptions.current.delete(id);
      setActiveJobs((list) => list.filter((j) => j !== id));
      setLocal((map) => Object.fromEntries(Object.entries(map).filter(([, l]) => l.jobId !== id)));
      refresh();
    },
    [refresh],
  );

  const onEvent = useCallback(
    (jobId: string, event: PrepareEvent) => {
      if (event.event === "model" && typeof event.stage === "string") {
        const k = keyOf({ stage: event.stage, index: typeof event.index === "number" ? event.index : null });
        const state = typeof event.state === "string" ? event.state : "preparing";
        const msg = typeof event.message === "string" ? event.message : null;
        if (ACTIVE.has(state)) {
          if (isReady(k)) return;
          setLocal((map) => ({ ...map, [k]: { items: map[k]?.items ?? {}, state, jobId } }));
          return;
        }
        // The model ended: remember a failure, forget a success or a cancel.
        setLocal((map) => {
          const next = { ...map };
          delete next[k];
          return next;
        });
        setAttempts((map) => {
          const next = { ...map };
          if (state === "ready" || state === "cancelled") delete next[k];
          else next[k] = { state, message: msg, ident: identOf({ type: String(event.type ?? ""), name: String(event.name ?? "") }) };
          return next;
        });
      } else if (event.event === "progress" && typeof event.stage === "string") {
        const k = keyOf({ stage: event.stage, index: typeof event.index === "number" ? event.index : null });
        const item = typeof event.item === "string" ? event.item : "";
        const done = typeof event.done === "number" ? event.done : 0;
        const total = typeof event.total === "number" ? event.total : null;
        if (isReady(k)) return;
        setLocal((map) => {
          const cur = map[k] ?? { state: "preparing", jobId, items: {} };
          return { ...map, [k]: { ...cur, state: "preparing", items: { ...cur.items, [item]: { done, total } } } };
        });
      } else if (event.event === "end") {
        if (event.status === "failed" && typeof event.error === "string") notify.current?.({ tone: "error", text: `Prepare failed: ${event.error}` });
        finishJob(jobId);
      }
    },
    [finishJob],
  );

  const follow = useCallback(
    (job: PrepareJob) => {
      setActiveJobs((list) => (list.includes(job.id) ? list : [...list, job.id]));
      setLocal((map) => {
        const next = { ...map };
        for (const m of job.models) if (!isReady(keyOf(m))) next[keyOf(m)] = { state: ACTIVE.has(m.state) ? m.state : "queued", jobId: job.id, items: {} };
        return next;
      });
      setAttempts((map) => {
        const next = { ...map };
        for (const m of job.models) delete next[keyOf(m)];
        return next;
      });
      const stop = api.prepareEvents(
        job.id,
        (event) => onEvent(job.id, event),
        () => finishJob(job.id), // stream lost: settle from a fresh status
      );
      subscriptions.current.set(job.id, stop);
    },
    [api, finishJob, onEvent],
  );

  const prepare = useCallback(
    async (models: ModelKey[] | "all") => {
      const cfg = configRef.current;
      if (!cfg) return;
      setSubmitting(true);
      try {
        const job = await api.prepare(models === "all" ? "all" : models.map((m) => ({ stage: m.stage, index: m.index })), cfg);
        follow(job);
        if (job.skipped.length) {
          const names = job.skipped.map((s) => `${where(s)} ${s.name}`).join(", ");
          notify.current?.({ tone: "warn", text: `Not prepared from the web (${plural(job.skipped.length, "model")}): ${names}. Unknown source: prepare it from the CLI (--prepare).` });
        }
      } catch (exc) {
        notify.current?.({ tone: "error", text: `Could not start Prepare: ${exc instanceof Error ? exc.message : String(exc)}` });
        refresh();
      } finally {
        setSubmitting(false);
      }
    },
    [api, follow, refresh],
  );

  const cancel = useCallback(async () => {
    await Promise.all(activeJobs.map((id) => api.cancelPrepare(id).catch(() => undefined)));
  }, [activeJobs, api]);

  // A Prepare started elsewhere (another tab, before a reload): poll until it ends.
  const foreignPreparing = activeJobs.length === 0 && rows.some((r) => r.state === "preparing");
  useEffect(() => {
    if (!foreignPreparing) return;
    const timer = setInterval(refresh, PREPARING_POLL_MS);
    return () => clearInterval(timer);
  }, [foreignPreparing, refresh]);

  useEffect(() => {
    const subs = subscriptions.current;
    return () => {
      subs.forEach((stop) => stop());
      subs.clear();
    };
  }, []);

  const views = useMemo<ModelView[]>(
    () =>
      rows.map((row) => {
        const k = keyOf(row);
        const mine = local[k];
        const tried = attempts[k];
        const attempt = tried && tried.ident === identOf(row) && row.state !== "ready" ? { state: tried.state, message: tried.message } : null;
        let state: ModelState = row.state;
        let msg = row.message;
        if (mine) state = mine.state === "queued" ? "queued" : "preparing";
        else if (attempt && row.state !== "preparing") {
          state = attempt.state === "failed" ? "failed" : row.state;
          msg = attempt.message ?? row.message;
        }
        return { ...row, state, message: msg, serverState: row.state, serverMessage: row.message, progress: mine ? progressOf(mine) : null, attempt };
      }),
    [attempts, local, rows],
  );

  return useMemo(() => {
    const notReady = views.filter((v) => v.state !== "ready");
    const toPrepare = notReady.filter((v) => preparable(v));
    // What Prepare all would download (a source the web may not fetch is not counted).
    const size = sizeOf(toPrepare);
    const preparing = views.filter((v) => v.state === "preparing" || v.state === "queued");
    return {
      status,
      message,
      rows: views,
      notReady,
      toPrepare,
      preparing,
      size,
      submitting,
      busy: activeJobs.length > 0,
      rowsFor: (stageKey: string, index: number | null) => rowsForSlot(views, stageKey, index),
      prepare,
      cancel,
      refresh,
    };
  }, [activeJobs.length, cancel, message, prepare, refresh, status, submitting, views]);
}

export type Readiness = ReturnType<typeof useModelStatus>;

const ReadinessContext = createContext<Readiness | null>(null);

export function ReadinessProvider({ value, children }: { value: Readiness; children: ReactNode }) {
  return createElement(ReadinessContext.Provider, { value }, children);
}

/** The pipeline's model readiness, or null outside a provider (then nothing is shown). */
export const useReadiness = (): Readiness | null => useContext(ReadinessContext);
