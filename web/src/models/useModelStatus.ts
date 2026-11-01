// Readiness of the edited pipeline's models: a debounced `POST
// /api/models/status` (a response to an older config is dropped: the
// sequence is bumped when a request is scheduled, as in the validation), the
// Prepare jobs of this server (the ones this page starts, and unfinished ones
// found on load or while the server reports "preparing": SSE progress per
// model), and the outcome of the last attempt per model (a failed Prepare
// stays "failed" with its reason after the refresh, until it is retried or the
// model becomes ready; a model that just became ready stays ready until the
// next status answer).
//
// Every overlay carries the identity (type + name) of the model it is about
// and applies to a row only when the row has that identity, and a row applies
// to a card only while the edited config still has that module at that place:
// removing or reordering modules never moves a state to another card.
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
  ident: string;
  items: Record<string, { done: number; total: number | null }>;
}

/** A model a job just made ready: shown ready until a status asked after it. */
interface Finished {
  ident: string;
  /** Status sequence when it finished: only a later request clears it. */
  seq: number;
}

interface Attempt {
  state: ModelState;
  message: string | null;
  /** Identity of the model it was about (a changed type or name drops it). */
  ident: string;
}

const identOf = (r: object): string => {
  const { type, name } = r as { type?: unknown; name?: unknown };
  return `${String(type ?? "")}\u0000${String(name ?? "")}`;
};

const keyOfEvent = (event: PrepareEvent) => keyOf({ stage: String(event.stage), index: typeof event.index === "number" ? event.index : null });

const isRecord = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v);

/**
 * Does the edited config still hold this row's module at the row's place?
 * A row of a key the config sets is checked against the spec there (type,
 * and name when the spec sets one; a list index past the end fails); a key
 * the config does not set (server defaults, sub-paths) cannot be checked and
 * is kept.
 */
export function rowMatchesConfig(row: ModelStatus, config: ConfigDict | null): boolean {
  if (!config || !(row.stage in config)) return true;
  const slot = config[row.stage];
  const spec = row.index === null ? slot : Array.isArray(slot) ? slot[row.index] : undefined;
  if (spec === undefined || spec === null) return false;
  if (!isRecord(spec)) return true;
  if (typeof spec.type === "string" && spec.type !== row.type) return false;
  if (typeof spec.name === "string" && spec.name !== row.name) return false;
  return true;
}

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

/**
 * The preparing overlay of a row: the one at its key when it is about this
 * module, else the only one of the same stage about this module (it moved:
 * the job keeps the index it was submitted with).
 */
function overlayFor(local: Record<string, Local>, row: ModelStatus, k: string, ident: string): Local | undefined {
  if (local[k]?.ident === ident) return local[k];
  const prefix = `${row.stage}|`;
  const same = Object.entries(local).filter(([key, l]) => key.startsWith(prefix) && l.ident === ident);
  return same.length === 1 ? same[0][1] : undefined;
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
  const [local, setLocal] = useState<Record<string, Local>>({});
  const [attempts, setAttempts] = useState<Record<string, Attempt>>({});
  const [finished, setFinished] = useState<Record<string, Finished>>({});
  const [activeJobs, setActiveJobs] = useState<string[]>([]);
  const activeRef = useRef(activeJobs);
  activeRef.current = activeJobs;
  const [submitting, setSubmitting] = useState(false);
  const [tick, setTick] = useState(0);
  const seq = useRef(0);
  const subscriptions = useRef(new Map<string, () => void>());
  // Models whose final event was seen, per job: a replayed stream (EventSource
  // reconnect) never takes them back to preparing.
  const ended = useRef(new Map<string, Set<string>>());

  const refresh = useCallback(() => setTick((n) => n + 1), []);
  // A model already ready is a no-op for the server: never shown as preparing.
  const isReady = useCallback((k: string, ident: string) => rowsRef.current.some((r) => keyOf(r) === k && identOf(r) === ident && r.state === "ready"), []);

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
          // Asked after those models finished: the server now knows.
          setFinished((map) => Object.fromEntries(Object.entries(map).filter(([, f]) => f.seq >= mine)));
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
      ended.current.delete(id);
      setActiveJobs((list) => list.filter((j) => j !== id));
      setLocal((map) => Object.fromEntries(Object.entries(map).filter(([, l]) => l.jobId !== id)));
      refresh();
    },
    [refresh],
  );

  const onEvent = useCallback(
    (jobId: string, event: PrepareEvent) => {
      if (event.event === "model" && typeof event.stage === "string") {
        const k = keyOfEvent(event);
        const ident = identOf(event);
        const state = typeof event.state === "string" ? event.state : "preparing";
        const msg = typeof event.message === "string" ? event.message : null;
        const seen = ended.current.get(jobId) ?? new Set<string>();
        ended.current.set(jobId, seen);
        if (ACTIVE.has(state)) {
          if (seen.has(k) || isReady(k, ident)) return; // a replay, or a no-op
          setLocal((map) => {
            const cur = map[k]?.ident === ident ? map[k] : undefined;
            // a replayed "queued" never takes a running model back
            const next = cur?.state === "preparing" && state === "queued" ? "preparing" : state;
            return { ...map, [k]: { items: cur?.items ?? {}, state: next, jobId, ident } };
          });
          return;
        }
        // The model ended: a success shows ready until the next status
        // answer; a failure is remembered; a cancel is forgotten.
        seen.add(k);
        setLocal((map) => {
          if (!map[k]) return map;
          const next = { ...map };
          delete next[k];
          return next;
        });
        if (state === "ready") setFinished((map) => ({ ...map, [k]: { ident, seq: seq.current } }));
        setAttempts((map) => {
          const next = { ...map };
          if (state === "ready" || state === "cancelled") delete next[k];
          else next[k] = { state, message: msg, ident };
          return next;
        });
      } else if (event.event === "progress" && typeof event.stage === "string") {
        const k = keyOfEvent(event);
        if (ended.current.get(jobId)?.has(k)) return; // replayed after the model ended
        const item = typeof event.item === "string" ? event.item : "";
        const done = typeof event.done === "number" ? event.done : 0;
        const total = typeof event.total === "number" ? event.total : null;
        setLocal((map) => {
          const cur = map[k];
          if (!cur || cur.jobId !== jobId) return map; // only models this job announced
          const prev = cur.items[item];
          // never rewind: a reconnect replays the stream from the start
          const kept = { done: Math.max(prev?.done ?? 0, done), total: total ?? prev?.total ?? null };
          return { ...map, [k]: { ...cur, state: "preparing", items: { ...cur.items, [item]: kept } } };
        });
      } else if (event.event === "end") {
        if (event.status === "failed" && typeof event.error === "string") notify.current?.({ tone: "error", text: `Prepare failed: ${event.error}` });
        finishJob(jobId);
      }
    },
    [finishJob, isReady],
  );

  const follow = useCallback(
    (job: PrepareJob) => {
      if (subscriptions.current.has(job.id)) return;
      setActiveJobs((list) => (list.includes(job.id) ? list : [...list, job.id]));
      const waiting = job.models.filter((m) => m.state === "queued" || m.state === "preparing");
      setLocal((map) => {
        const next = { ...map };
        for (const m of waiting) {
          const ident = identOf(m);
          if (!isReady(keyOf(m), ident)) next[keyOf(m)] = { state: m.state, jobId: job.id, ident, items: {} };
        }
        return next;
      });
      setAttempts((map) => {
        const next = { ...map };
        for (const m of waiting) delete next[keyOf(m)];
        return next;
      });
      const stop = api.prepareEvents(
        job.id,
        (event) => onEvent(job.id, event),
        () => finishJob(job.id), // stream lost: settle from a fresh status
      );
      subscriptions.current.set(job.id, stop);
    },
    [api, finishJob, isReady, onEvent],
  );

  /** Re-attach to the server's unfinished jobs (after a reload, or started elsewhere). */
  const attach = useCallback(async () => {
    let jobs: PrepareJob[];
    try {
      jobs = await api.listPrepares();
    } catch {
      refresh(); // an older server without the listing: just poll the status
      return;
    }
    const open = jobs.filter((j) => j.status === "queued" || j.status === "running");
    open.forEach(follow);
    if (!open.length) refresh();
  }, [api, follow, refresh]);

  useEffect(() => {
    void attach();
  }, [attach]);

  const prepare = useCallback(
    async (models: ModelKey[] | "all") => {
      const cfg = configRef.current;
      if (!cfg) return;
      setSubmitting(true);
      try {
        const job = await api.prepare(models === "all" ? "all" : models.map((m) => ({ stage: m.stage, index: m.index })), cfg);
        follow({ ...job, models: job.models.map((m) => ({ ...m, state: m.state === "preparing" ? m.state : "queued" })) });
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
    await Promise.all(activeRef.current.map((id) => api.cancelPrepare(id).catch(() => undefined)));
  }, [api]);

  // The server reports "preparing" for a job this page does not follow: find it.
  const foreignPreparing = activeJobs.length === 0 && rows.some((r) => r.state === "preparing");
  useEffect(() => {
    if (!foreignPreparing) return;
    const timer = setInterval(() => void attach(), PREPARING_POLL_MS);
    return () => clearInterval(timer);
  }, [attach, foreignPreparing]);

  useEffect(() => {
    const subs = subscriptions.current;
    return () => {
      subs.forEach((stop) => stop());
      subs.clear();
    };
  }, []);

  const views = useMemo<ModelView[]>(
    () =>
      rows
        .filter((row) => rowMatchesConfig(row, configRef.current))
        .map((row) => {
          const k = keyOf(row);
          const ident = identOf(row);
          const found = overlayFor(local, row, k, ident);
          // queued in a job but already ready (a "Prepare all" no-op, or attached before the rows came)
          const mine = found && !(found.state === "queued" && row.state === "ready") ? found : undefined;
          const done = finished[k]?.ident === ident;
          const tried = attempts[k];
          const attempt = tried && tried.ident === ident && row.state !== "ready" && !done ? { state: tried.state, message: tried.message } : null;
          let state: ModelState = row.state;
          let msg = row.message;
          let preparableNow = row.preparable;
          if (mine) state = mine.state === "queued" ? "queued" : "preparing";
          else if (done && row.state !== "ready") {
            state = "ready";
            msg = null;
            preparableNow = false;
          } else if (attempt && row.state !== "preparing") {
            state = attempt.state === "failed" ? "failed" : row.state;
            msg = attempt.message ?? row.message;
          }
          return { ...row, state, message: msg, preparable: preparableNow, serverState: row.state, serverMessage: row.message, progress: mine ? progressOf(mine) : null, attempt };
        }),
    [attempts, configText, finished, local, rows], // configText: the content of configRef

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
