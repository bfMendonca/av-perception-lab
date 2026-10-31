// Jobs submitted from this page: queue position, progress against the
// expected frame count, the latest annotated frame (preview.jpg), Cancel,
// a link to the run when done and the log tail when it failed.
import { useEffect, useState } from "react";
import { Icon } from "../ui/icons";
import { isActive, type SessionJob } from "./useJobs";

export interface JobPanelProps {
  jobs: SessionJob[];
  previewUrl: (runId: string) => string;
  onCancel: (id: string) => void;
  onOpenRun: (runId: string) => void;
  onDismiss: (id: string) => void;
}

const STATUS_LABEL: Record<string, string> = {
  queued: "Queued",
  running: "Running",
  done: "Done",
  failed: "Failed",
  cancelled: "Cancelled",
};

export function JobPanel({ jobs, previewUrl, onCancel, onOpenRun, onDismiss }: JobPanelProps) {
  if (!jobs.length) {
    return (
      <div className="jobs jobs--empty">
        <Icon name="spark" size={20} />
        <p>Processed frames appear here as they arrive.</p>
        <p className="muted">Runs queue one at a time on the GPU.</p>
      </div>
    );
  }
  return (
    <ol className="jobs" aria-label="Jobs">
      {jobs.map((job, i) => (
        <JobItem key={job.id} job={job} expanded={i === 0 || isActive(job)} previewUrl={previewUrl} onCancel={onCancel} onOpenRun={onOpenRun} onDismiss={onDismiss} />
      ))}
    </ol>
  );
}

function JobItem({ job, expanded, previewUrl, onCancel, onOpenRun, onDismiss }: { job: SessionJob; expanded: boolean } & Omit<JobPanelProps, "jobs">) {
  const active = isActive(job);
  const ratio = job.expected > 0 ? Math.min(job.frames / job.expected, 1) : 0;
  const statusText =
    job.status === "queued"
      ? job.position === null
        ? "Queued"
        : job.position === 0
          ? "Queued · starts next"
          : `Queued · ${job.position} ahead`
      : job.status === "running" && job.frames === 0
        ? "Loading models…"
        : (STATUS_LABEL[job.status] ?? job.status);
  return (
    <li className={`job job--${job.status}`} aria-label={`Job ${job.label}`}>
      <div className="job__head">
        <span className={`job__dot job__dot--${job.status}`} aria-hidden="true" />
        <div className="job__titles">
          <span className="job__label" title={job.label}>
            {job.label}
          </span>
          <span className="job__status" role="status">
            {statusText}
          </span>
        </div>
        {active ? (
          <button type="button" className="btn btn--xs btn--danger" onClick={() => onCancel(job.id)}>
            <Icon name="stop" size={12} /> Cancel
          </button>
        ) : (
          <button type="button" className="icon-btn icon-btn--xs" aria-label="Dismiss" title="Dismiss" onClick={() => onDismiss(job.id)}>
            <Icon name="x" />
          </button>
        )}
      </div>
      {job.status === "running" || (job.status !== "queued" && job.frames > 0) ? (
        <div className="job__progress">
          <div
            className={`progress${job.expected > 0 ? "" : " progress--indeterminate"}`}
            role="progressbar"
            aria-label="Frames processed"
            aria-valuemin={0}
            aria-valuemax={job.expected || undefined}
            aria-valuenow={job.frames}
          >
            <span style={{ width: `${(job.status === "done" ? 1 : ratio) * 100}%` }} />
          </div>
          <span className="job__count">
            {job.frames}
            {job.expected > 0 ? ` / ${job.expected}` : ""} frames
          </span>
        </div>
      ) : null}
      {job.status === "failed" ? (
        <div className="job__failure">
          {job.error ? <p className="job__error">{job.error}</p> : null}
          {job.logTail.length ? (
            <details open={!job.error}>
              <summary>Log tail ({job.logTail.length} lines)</summary>
              <pre className="job__log">{job.logTail.join("\n")}</pre>
            </details>
          ) : null}
        </div>
      ) : null}
      {expanded && job.runId && job.previewTick && job.frames > 0 ? <LivePreview src={`${previewUrl(job.runId)}?v=${job.previewTick}`} live={active} /> : null}
      {job.status === "done" && job.runId ? (
        <button type="button" className="btn btn--sm job__open" onClick={() => onOpenRun(job.runId as string)}>
          Open run <Icon name="chevron" size={13} />
        </button>
      ) : null}
    </li>
  );
}

/** Swaps to a new frame only once it has loaded (no flicker, a 404 keeps the last one). */
function LivePreview({ src, live }: { src: string; live: boolean }) {
  const [shown, setShown] = useState<string | null>(null);
  useEffect(() => {
    let current = true;
    const img = new Image();
    img.onload = () => current && setShown(src);
    img.src = src;
    return () => {
      current = false;
    };
  }, [src]);
  if (!shown) return live ? <div className="job__frame job__frame--waiting" aria-hidden="true" /> : null;
  return (
    <figure className="job__frame">
      <img src={shown} alt="Latest annotated frame" />
      {live ? <figcaption>Live</figcaption> : null}
    </figure>
  );
}
