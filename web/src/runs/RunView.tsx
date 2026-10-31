// Minimal runs view: the history list and one run (annotated video, files).
// It is where a finished job lands; the full results and comparison view
// replaces it in the next delivery.
import { useEffect, useState } from "react";
import type { RunRecord, VideoApi } from "../api/video";
import { Icon } from "../ui/icons";

export interface RunViewProps {
  api: VideoApi;
  runId: string | null;
  onSelect: (runId: string) => void;
}

const FILES: { name: string; label: string }[] = [
  { name: "video", label: "Annotated video" },
  { name: "config", label: "config.yaml" },
  { name: "detections", label: "detections.csv" },
  { name: "preview", label: "Last frame" },
];

function when(iso: string): string {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString("en-GB", { dateStyle: "medium", timeStyle: "short" });
}

export function RunView({ api, runId, onSelect }: RunViewProps) {
  const [runs, setRuns] = useState<RunRecord[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [run, setRun] = useState<RunRecord | null>(null);

  useEffect(() => {
    let live = true;
    api
      .listRuns()
      .then((r) => live && setRuns(r))
      .catch((exc: unknown) => live && setError(exc instanceof Error ? exc.message : String(exc)));
    return () => {
      live = false;
    };
  }, [api, runId]);

  useEffect(() => {
    let live = true;
    setRun(null);
    if (runId)
      api
        .getRun(runId)
        .then((r) => live && setRun(r))
        .catch(() => live && setRun(null));
    return () => {
      live = false;
    };
  }, [api, runId]);

  return (
    <div className="runs">
      <aside className="runs__list" aria-label="Run history">
        <h3>History</h3>
        {error ? <p className="muted">Could not load the history: {error}</p> : null}
        {runs === null && !error ? <p className="muted">Loading…</p> : null}
        {runs?.length === 0 ? <p className="muted">No runs yet. Process a video segment from the editor.</p> : null}
        <ul>
          {runs?.map((r) => (
            <li key={r.id}>
              <button type="button" className={`runitem${r.id === runId ? " runitem--on" : ""}`} aria-current={r.id === runId ? "true" : undefined} onClick={() => onSelect(r.id)}>
                <span className="runitem__name" title={r.id}>
                  {r.video ?? r.id}
                </span>
                <span className="runitem__meta">
                  <span className={`job__dot job__dot--${r.status}`} aria-hidden="true" /> {r.status} · {when(r.created)}
                  {r.frames ? ` · ${r.frames} frames` : ""}
                </span>
              </button>
            </li>
          ))}
        </ul>
      </aside>
      <section className="runs__detail" aria-label="Run">
        {!runId ? (
          <div className="state-panel">
            <Icon name="history" size={28} />
            <h2>Pick a run</h2>
            <p>Finished jobs open here.</p>
          </div>
        ) : run ? (
          <>
            <header className="runs__head">
              <div>
                <h2>{run.video ?? run.id}</h2>
                <p className="muted">
                  {run.video ? (
                    <>
                      <code>{run.id}</code> ·{" "}
                    </>
                  ) : null}
                  {run.status}
                  {run.frames ? ` · ${run.frames} frames` : ""}
                  {run.wall_s ? ` · ${run.wall_s.toFixed(1)} s` : ""}
                  {run.detectors.length ? ` · ${run.detectors.join(" + ")}` : ""}
                  {run.depth ? ` · ${run.depth}` : ""}
                </p>
              </div>
            </header>
            {run.error ? <p className="job__error">{run.error}</p> : null}
            <video className="runs__video" src={api.runFileUrl(run.id, "video")} controls autoPlay muted playsInline poster={api.runFileUrl(run.id, "preview")} />
            <ul className="runs__files">
              {FILES.map((f) => (
                <li key={f.name}>
                  <a className="btn btn--sm" href={api.runFileUrl(run.id, f.name)} target="_blank" rel="noreferrer">
                    <Icon name="file" size={13} /> {f.label}
                  </a>
                </li>
              ))}
            </ul>
          </>
        ) : (
          <p className="muted runs__loading">Loading {runId}…</p>
        )}
      </section>
    </div>
  );
}
