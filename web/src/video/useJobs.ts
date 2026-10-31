// Jobs of this page: status from the job's SSE stream, queue position polled
// while it waits, and a throttled tick that refreshes the live preview.jpg on
// `frame` events. On mount the server's unfinished jobs are followed again
// (a reload, or another browser), and a job the server no longer knows (it
// restarted) is settled instead of staying "running".
import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError } from "../api/client";
import type { ConfigDict } from "../api/types";
import type { JobEvent, JobInfo, JobStatus, VideoApi } from "../api/video";

const POSITION_POLL_MS = 1500;
const PREVIEW_REFRESH_MS = 600;
const TERMINAL: JobStatus[] = ["done", "failed", "cancelled"];
export const LOST_ERROR = "the server no longer knows this job (it restarted): see Runs";

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
      } catch (exc) {
        // The server restarted (404): the job cannot still be running.
        if (exc instanceof ApiError && exc.status === 404) {
          patch(id, (j) => (isActive(j) ? { status: "failed", position: null, error: j.error ?? LOST_ERROR } : {}));
        }
      }
    },
    [api, patch],
  );

  const onEvent = useCallback(
    (id: string, event: JobEvent) => {
      switch (event.event) {
        case "queued":
          // a replay from the start never moves a running job back
          patch(id, (j) => (j.status === "running" ? {} : { status: "queued" }));
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
          patch(id, (j) => ({ frames: typeof event.n === "number" ? Math.max(j.frames, event.n) : j.frames + 1, status: "running", position: null, ...(fresh ? { previewTick: now } : {}) }));
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

  const known = useRef(new Set<string>());

  /** Follow a job queued elsewhere (e.g. a rerun from the runs view, or one the server lists after a reload). */
  const track = useCallback(
    (id: string, meta: { label: string; expected: number }, info?: JobInfo) => {
      if (known.current.has(id) || jobsRef.current.some((j) => j.id === id)) return;
      known.current.add(id);
      const job: SessionJob = {
        id,
        label: meta.label,
        status: info?.status ?? "queued",
        position: info?.position ?? null,
        frames: info?.frames ?? 0,
        expected: meta.expected,
        runId: info?.run_id ?? null,
        error: info?.error ?? null,
        logTail: info?.log_tail ?? [],
        previewTick: info?.run_id && info.frames > 0 ? Date.now() : 0,
        submittedAt: info?.created ? Date.parse(info.created) || Date.now() : Date.now(),
      };
      setJobs((list) => [job, ...list]);
      subscribe(id);
    },
    [subscribe],
  );

  // A reloaded page (or another browser) follows the server's unfinished jobs.
  useEffect(() => {
    let live = true;
    api
      .listJobs()
      .then((list) => {
        if (!live) return;
        [...list].reverse().forEach((info) => {
          if (TERMINAL.includes(info.status)) return;
          track(info.id, { label: info.video ?? `job ${info.id}`, expected: 0 }, info);
        });
      })
      .catch(() => undefined); // an older server without the listing
    return () => {
      live = false;
    };
  }, [api, track]);

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
