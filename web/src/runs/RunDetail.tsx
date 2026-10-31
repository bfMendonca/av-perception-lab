// One run: annotated video, summary with deltas against the comparison run,
// configuration diff, distance-over-time chart (click seeks the video),
// discovered artifacts and the generic metrics tree. A `job-<id>` entry (a
// failed job without a folder) shows its error and log instead.
import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import type { Artifact, Comparison, RunsApi, Series, SummaryResponse } from "../api/runs";
import type { RunRecord } from "../api/video";
import { Icon } from "../ui/icons";
import { Artifacts } from "./Artifacts";
import { artifactViewFor } from "./artifactViews";
import { ConfigDiff, KpiCards } from "./CompareView";
import { DistanceChart } from "./DistanceChart";
import { formatSeconds, formatWhen } from "./format";
import { hasFiles, runTitle } from "./RunList";

export interface RunDetailProps {
  run: RunRecord;
  /** The comparison run (A); the shown run is B. */
  against: RunRecord | null;
  runs: RunRecord[];
  api: RunsApi;
  onCompare: (id: string | null) => void;
  onReopen: (run: RunRecord) => Promise<void>;
  onRerun: (run: RunRecord) => Promise<void>;
}

type Load<T> = { status: "loading" } | { status: "error"; message: string } | { status: "ready"; value: T };

const message = (exc: unknown) => (exc instanceof Error ? exc.message : String(exc));

function useLoad<T>(load: (() => Promise<T>) | null, deps: unknown[]): Load<T> | null {
  const [state, setState] = useState<Load<T> | null>(null);
  useEffect(() => {
    if (!load) {
      setState(null);
      return;
    }
    let live = true;
    setState((prev) => (prev && prev.status === "ready" ? prev : { status: "loading" }));
    load()
      .then((value) => live && setState({ status: "ready", value }))
      .catch((exc: unknown) => live && setState({ status: "error", message: message(exc) }));
    return () => {
      live = false;
    };
  }, deps);
  return state;
}

function Actions({ run, onReopen, onRerun }: Pick<RunDetailProps, "run" | "onReopen" | "onRerun">) {
  const [busy, setBusy] = useState<"reopen" | "rerun" | null>(null);
  const act = async (kind: "reopen" | "rerun") => {
    setBusy(kind);
    try {
      await (kind === "reopen" ? onReopen(run) : onRerun(run));
    } finally {
      setBusy(null);
    }
  };
  const why = run.has_config ? undefined : "This run recorded no configuration";
  return (
    <div className="runs__actions">
      <button type="button" className="btn" disabled={!run.has_config || busy !== null} title={why ?? "Load this run's configuration into the editor"} onClick={() => void act("reopen")}>
        <Icon name="layers" size={14} /> {busy === "reopen" ? "Opening…" : "Reopen in editor"}
      </button>
      <button type="button" className="btn btn--primary" disabled={!run.has_config || busy !== null} title={why ?? "Queue the same configuration and video again"} onClick={() => void act("rerun")}>
        <Icon name="reset" size={14} /> {busy === "rerun" ? "Queueing…" : "Rerun"}
      </button>
    </div>
  );
}

function Header({ run, children }: { run: RunRecord; children?: ReactNode }) {
  return (
    <header className="runs__head">
      <div className="runs__title">
        <h2>
          {runTitle(run)}
          <span className={`status-pill status-pill--${run.status}`}>{run.status}</span>
        </h2>
        <p className="muted">
          <code title="Run id">{run.id}</code>
          {" · "}
          {formatWhen(run.created)}
          {run.frames ? ` · ${run.frames} frames` : ""}
          {run.wall_s ? ` · ${formatSeconds(run.wall_s)} wall` : ""}
          {run.detectors.length ? ` · ${run.detectors.join(" + ")}` : ""}
          {run.depth ? ` · ${run.depth}` : ""}
        </p>
      </div>
      {children}
    </header>
  );
}

export function RunDetail(props: RunDetailProps) {
  const { run, against } = props;
  if (!hasFiles(run)) {
    return (
      <div className="runs__detail-inner">
        <Header run={run}>
          <Actions run={run} onReopen={props.onReopen} onRerun={props.onRerun} />
        </Header>
        <section className="rcard rcard--failure" aria-label="Why the job failed">
          <h3>
            <Icon name="alert" size={14} /> The job {run.status} before writing any file
          </h3>
          {run.error ? <p className="job__error">{run.error}</p> : null}
          {run.log_tail?.length ? (
            <pre className="runs__log" tabIndex={0} aria-label="Log tail">
              {run.log_tail.join("\n")}
            </pre>
          ) : (
            <p className="muted">No log was captured.</p>
          )}
        </section>
      </div>
    );
  }
  return <RunResults {...props} key={run.id} against={against && hasFiles(against) ? against : null} />;
}

function RunResults({ run, against, runs, api, onCompare, onReopen, onRerun }: RunDetailProps) {
  const video = useRef<HTMLVideoElement>(null);
  const [playhead, setPlayhead] = useState<number | null>(null);
  const [track, setTrack] = useState<number | null>(null);
  const againstId = against?.id ?? null;

  const artifacts = useLoad<Artifact[]>(() => api.artifacts(run.id), [api, run.id, run.status]);
  const single = useLoad<SummaryResponse>(againstId ? null : () => api.summary(run.id), [api, run.id, againstId, run.status]);
  const comparison = useLoad<Comparison>(againstId ? () => api.compare(againstId, run.id) : null, [api, run.id, againstId, run.status]);
  const seriesB = useLoad<Series>(() => api.series(run.id, track), [api, run.id, track, run.status]);
  const resolvedTrack = seriesB?.status === "ready" ? seriesB.value.track : null;
  const seriesA = useLoad<Series>(againstId && seriesB?.status === "ready" ? () => api.series(againstId, resolvedTrack) : null, [api, againstId, resolvedTrack, seriesB?.status]);
  // Track ids are per run: until the user picks one, show the longest track
  // of this run that the comparison run has too.
  const [picked, setPicked] = useState(false);
  const idsA = useMemo(() => new Set(seriesA?.status === "ready" ? seriesA.value.tracks.map((t) => t.id) : []), [seriesA]);
  useEffect(() => {
    if (picked || seriesB?.status !== "ready" || seriesA?.status !== "ready" || seriesA.value.found) return;
    const shared = seriesB.value.tracks.find((t) => idsA.has(t.id));
    if (shared && shared.id !== resolvedTrack) setTrack(shared.id);
  }, [idsA, picked, resolvedTrack, seriesA, seriesB]);

  const list = artifacts?.status === "ready" ? artifacts.value : [];
  const videoArt = list.find((a) => a.role === "video");
  const previewArt = list.find((a) => a.role === "preview");
  const metricsArt = list.find((a) => a.role === "metrics");
  const MetricsView = metricsArt ? artifactViewFor(metricsArt) : undefined;

  const seek = useCallback((t: number) => {
    const el = video.current;
    if (!el) return;
    el.currentTime = t;
    setPlayhead(t);
  }, []);

  const candidates = useMemo(() => runs.filter((r) => r.id !== run.id && hasFiles(r)), [runs, run.id]);
  const summary = againstId ? (comparison?.status === "ready" ? comparison.value.summary_b : null) : single?.status === "ready" ? single.value.summary : null;
  const metrics = againstId ? (comparison?.status === "ready" ? comparison.value.metrics : []) : single?.status === "ready" ? single.value.metrics : [];
  const summaryState = againstId ? comparison : single;
  const tracks = seriesB?.status === "ready" ? seriesB.value.tracks : [];
  const labelB = "This run";
  const labelA = against ? `vs ${runTitle(against)}` : undefined;

  return (
    <div className="runs__detail-inner">
      <Header run={run}>
        <Actions run={run} onReopen={onReopen} onRerun={onRerun} />
      </Header>
      <div className="runs__grid">
        <section className="rcard rcard--video" aria-label="Annotated video">
          <h3>Annotated video</h3>
          {artifacts?.status === "loading" ? (
            <span className="skel runs__video-skel" aria-busy="true" />
          ) : videoArt ? (
            <video
              ref={video}
              className="runs__video"
              src={videoArt.url}
              poster={previewArt?.url}
              controls
              muted
              playsInline
              preload="metadata"
              onTimeUpdate={(e) => setPlayhead(e.currentTarget.currentTime)}
              onSeeked={(e) => setPlayhead(e.currentTarget.currentTime)}
            />
          ) : (
            <div className="runs__novideo">
              <Icon name="film" size={24} />
              <p className="muted">No annotated video in this run folder{run.status === "running" ? " yet (the run is in progress)" : ""}.</p>
            </div>
          )}
        </section>
        <section className="rcard rcard--summary" aria-label="Summary">
          <div className="rcard__head">
            <h3>{against ? "Summary · this run vs comparison" : "Summary"}</h3>
            <label className="cmp-pick">
              <span className="muted">Compare with</span>
              <select className="input select__el" value={againstId ?? ""} onChange={(e) => onCompare(e.target.value || null)} aria-label="Comparison run">
                <option value="">None</option>
                {candidates.map((r) => (
                  <option key={r.id} value={r.id}>
                    {runTitle(r)} · {formatWhen(r.created)}
                  </option>
                ))}
              </select>
            </label>
          </div>
          {summaryState?.status === "error" ? <p className="job__error">{summaryState.message}</p> : null}
          {summary ? (
            <>
              <KpiCards metrics={metrics} summary={summary} deltas={comparison?.status === "ready" ? comparison.value.deltas : null} comparing={Boolean(against)} />
              {summary.notes?.length ? (
                <ul className="runs__notes">
                  {summary.notes.map((n) => (
                    <li key={n}>
                      <Icon name="info" size={12} /> {n}
                    </li>
                  ))}
                </ul>
              ) : null}
            </>
          ) : summaryState?.status !== "error" ? (
            <div className="kpis kpis--loading" aria-busy="true">
              {[0, 1, 2].map((i) => (
                <span className="skel skel--kpi" key={i} />
              ))}
            </div>
          ) : null}
          <h3 className="rcard__sub">Configuration changes</h3>
          {!against ? (
            <p className="muted diff__empty">Pick a run to compare with (the list's compare button or the menu above) to see its configuration diff and deltas.</p>
          ) : comparison?.status === "ready" ? (
            <ConfigDiff diff={comparison.value.config_diff} origin={comparison.value.config_origin} />
          ) : comparison?.status === "loading" ? (
            <p className="muted">Comparing…</p>
          ) : null}
        </section>
        <section className="rcard rcard--wide" aria-label="Distance over time">
          <div className="rcard__head">
            <h3>Distance over time</h3>
            {tracks.length ? (
              <label className="cmp-pick">
                <span className="muted">Track</span>
                <select
                  className="input select__el"
                  value={resolvedTrack ?? ""}
                  onChange={(e) => {
                    setPicked(true);
                    setTrack(Number(e.target.value));
                  }}
                  aria-label="Track"
                >
                  {tracks.slice(0, 400).map((t) => (
                    <option key={t.id} value={t.id}>
                      #{t.id}
                      {t.class_name ? ` · ${t.class_name}` : ""} · {t.rows} pts · {formatSeconds(t.t0)}–{formatSeconds(t.t1)}
                      {against && idsA.has(t.id) ? " · in both runs" : ""}
                    </option>
                  ))}
                </select>
              </label>
            ) : seriesB?.status === "ready" ? (
              <span className="muted">every box (this run has no tracks)</span>
            ) : null}
          </div>
          {seriesB?.status === "error" ? <p className="job__error">{seriesB.message}</p> : null}
          {seriesB?.status === "ready" ? (
            <>
              <DistanceChart b={seriesB.value} a={seriesA?.status === "ready" ? seriesA.value : null} labelB={labelB} labelA={labelA} onSeek={seek} playhead={playhead} />
              {against && seriesA?.status === "ready" && !seriesA.value.found ? <p className="muted runs__notes">Track #{resolvedTrack} does not exist in the comparison run.</p> : null}
              {seriesA?.status === "error" ? <p className="muted runs__notes">Comparison series unavailable: {seriesA.message}</p> : null}
            </>
          ) : seriesB?.status === "loading" ? (
            <span className="skel skel--chart" aria-busy="true" />
          ) : null}
        </section>
        <section className="rcard" aria-label="Artifacts">
          <h3>Artifacts</h3>
          {artifacts?.status === "error" ? <p className="job__error">{artifacts.message}</p> : null}
          {artifacts?.status === "ready" ? <Artifacts artifacts={artifacts.value} api={api} /> : null}
        </section>
        <section className="rcard" aria-label="Metrics">
          <h3>Metrics</h3>
          {metricsArt && MetricsView ? <MetricsView artifact={metricsArt} api={api} /> : artifacts?.status === "ready" ? <p className="muted">No metrics.json (the run did not finish, or predates it).</p> : null}
        </section>
      </div>
    </div>
  );
}
