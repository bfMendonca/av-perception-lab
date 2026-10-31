// Jobs submitted from this page: status from the job's SSE stream, queue
// position polled while it waits, and a throttled tick that refreshes the
// live preview.jpg on `frame` events.
import { useCallback, useEffect, useRef, useState } from "react";
import type { ConfigDict } from "../api/types";
import type { JobEvent, JobStatus, VideoApi } from "../api/video";

const POSITION_POLL_MS = 1500;
const PREVIEW_REFRESH_MS = 600;
const TERMINAL: JobStatus[] = ["done", "failed", "cancelled"];

export interface SessionJob {
  id: string;
  /** What runs, e.g. "road.mp4 · 0:12.0–0:15.0". */
  label: string;
  status: JobStatus;
  position: number | null;
  frames: number;
  /** Frames the segment should yield (0 = unknown). */
  expected: number;
  runId: string | null;
  error: string | null;
  logTail: string[];
  /** Changes when a newer preview.jpg is worth fetching (0 = none yet). */
  previewTick: number;
  submittedAt: number;
}

export interface JobsOptions {
  /** A job finished with `done` (jump to its run). */
  onDone?: (runId: string, job: SessionJob) => void;
}

const baseName = (path: unknown) => (typeof path === "string" && path ? path.split(/[\\/]/).filter(Boolean).pop() ?? null : null);

export function isActive(job: SessionJob): boolean {
  return !TERMINAL.includes(job.status);
}

export function useJobs(api: VideoApi, options: JobsOptions = {}) {
  const [jobs, setJobs] = useState<SessionJob[]>([]);
  const jobsRef = useRef(jobs);
  jobsRef.current = jobs;
  const onDone = useRef(options.onDone);
  onDone.current = options.onDone;
  const subscriptions = useRef(new Map<string, () => void>());
  const lastTick = useRef(new Map<string, number>());

  const patch = useCallback((id: string, update: (job: SessionJob) => Partial<SessionJob>) => {
    setJobs((list) => list.map((j) => (j.id === id ? { ...j, ...update(j) } : j)));
  }, []);

  const finish = useCallback(
    async (id: string) => {
      subscriptions.current.get(id)?.();
      subscriptions.current.delete(id);
      try {
        const info = await api.getJob(id);
        const settle = (j: SessionJob): SessionJob => ({
          ...j,
          status: info.status,
          position: null,
          error: info.error ?? j.error,
          runId: info.run_id ?? j.runId,
          frames: Math.max(j.frames, info.frames),
          logTail: info.log_tail,
          previewTick: j.previewTick || (info.run_id && info.frames > 0 ? Date.now() : 0),
        });
        setJobs((list) => list.map((j) => (j.id === id ? settle(j) : j)));
        const known = jobsRef.current.find((j) => j.id === id);
        const done = known ? settle(known) : null;
        if (done && done.status === "done" && done.runId) onDone.current?.(done.runId, done);
      } catch {
        // the job record is gone (server restart): keep what the stream said
      }
    },
    [api],
  );

  const onEvent = useCallback(
    (id: string, event: JobEvent) => {
      switch (event.event) {
        case "queued":
          patch(id, () => ({ status: "queued" }));
          break;
        case "running":
          patch(id, () => ({ status: "running", position: null }));
          break;
        case "started":
          patch(id, () => ({ runId: baseName(event.run_dir), status: "running", position: null }));
          break;
        case "frame": {
          const now = Date.now();
          const fresh = now - (lastTick.current.get(id) ?? 0) >= PREVIEW_REFRESH_MS;
          if (fresh) lastTick.current.set(id, now);
          patch(id, (j) => ({ frames: typeof event.n === "number" ? event.n : j.frames + 1, status: "running", position: null, ...(fresh ? { previewTick: now } : {}) }));
          break;
        }
        case "error":
          patch(id, () => ({ error: typeof event.message === "string" ? event.message : "the run failed" }));
          break;
        case "done":
          patch(id, (j) => ({ runId: baseName(event.run_dir) ?? j.runId, frames: typeof event.frames === "number" ? event.frames : j.frames }));
          break;
        case "end":
          patch(id, (j) => ({ status: typeof event.status === "string" ? event.status : j.status, error: typeof event.error === "string" ? event.error : j.error, position: null }));
          void finish(id);
          break;
        default:
          break; // events this page does not know are ignored
      }
    },
    [finish, patch],
  );

  const subscribe = useCallback(
    (id: string) => {
      const stop = api.jobEvents(
        id,
        (event) => onEvent(id, event),
        () => void finish(id), // stream lost: settle from the job record
      );
      subscriptions.current.set(id, stop);
    },
    [api, finish, onEvent],
  );

  /** Follow a job queued elsewhere (e.g. a rerun from the runs view). */
  const track = useCallback(
    (id: string, meta: { label: string; expected: number }) => {
      if (jobsRef.current.some((j) => j.id === id)) return;
      const job: SessionJob = {
        id,
        label: meta.label,
        status: "queued",
        position: null,
        frames: 0,
        expected: meta.expected,
        runId: null,
        error: null,
        logTail: [],
        previewTick: 0,
        submittedAt: Date.now(),
      };
      setJobs((list) => [job, ...list]);
      subscribe(id);
    },
    [subscribe],
  );

  const submit = useCallback(
    async (config: ConfigDict, videoId: string | undefined, meta: { label: string; expected: number }) => {
      const id = await api.submitRun(config, videoId);
      track(id, meta);
      return id;
    },
    [api, track],
  );

  const cancel = useCallback(
    async (id: string) => {
      try {
        const info = await api.cancelJob(id);
        patch(id, () => ({ status: info.status, position: null }));
      } catch {
        // already gone: the stream settles it
      }
    },
    [api, patch],
  );

  const dismiss = useCallback((id: string) => setJobs((list) => list.filter((j) => j.id !== id || isActive(j))), []);

  // Queue position is not an event: poll it while a job waits.
  const waiting = jobs.filter((j) => j.status === "queued").map((j) => j.id).join(",");
  useEffect(() => {
    if (!waiting) return;
    const ids = waiting.split(",");
    const poll = () =>
      ids.forEach((id) =>
        api
          .getJob(id)
          .then((info) => patch(id, (j) => (j.status === "queued" ? { position: info.position, status: info.status === "queued" ? "queued" : j.status } : {})))
          .catch(() => undefined),
      );
    poll();
    const timer = setInterval(poll, POSITION_POLL_MS);
    return () => clearInterval(timer);
  }, [api, patch, waiting]);

  useEffect(() => {
    const subs = subscriptions.current;
    return () => {
      subs.forEach((stop) => stop());
      subs.clear();
    };
  }, []);

  return { jobs, submit, track, cancel, dismiss };
}

export type Jobs = ReturnType<typeof useJobs>;
