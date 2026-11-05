// The video deck above the pipeline editor: intake (drop / browse / server
// path / library), the player (for an MCAP: its thumbnail and the topic
// picker), the segment bar with Process, and the jobs.
import { useRef, useState } from "react";
import type { McapTopic } from "../api/types";
import { isMcap, type VideoRecord } from "../api/video";
import { useEditor } from "../editor/yamlState";
import { Icon } from "../ui/icons";
import { DropZone, formatBytes, isMcapName, type DroppedFiles } from "./DropZone";
import { bindMcap, formatRate, formatResolution, inputTiming, setUseCameraInfo, sourceTopic, topicsOf, usesCameraInfo } from "./mcapCamera";
import { JobPanel } from "./JobPanel";
import { Player, type PlayerHandle } from "./Player";
import { SegmentBar } from "./SegmentBar";
import { TopicPicker } from "./TopicPicker";
import { formatDuration, formatTime, SOURCE_KEY } from "./segment";
import type { Jobs } from "./useJobs";
import type { Process } from "./useProcess";
import type { VideoSession } from "./useVideoSession";

const COLLAPSED_KEY = "ddd.deck.collapsed";

export interface VideoDeckProps {
  session: VideoSession;
  jobs: Jobs;
  process: Process;
  previewUrl: (runId: string) => string;
  onFiles: (files: DroppedFiles) => void;
  onOpenRun: (runId: string) => void;
}

function storedCollapsed(): boolean {
  try {
    return window.localStorage.getItem(COLLAPSED_KEY) === "1";
  } catch {
    return false;
  }
}

export function VideoDeck({ session, jobs, process, previewUrl, onFiles, onOpenRun }: VideoDeckProps) {
  const editor = useEditor();
  const player = useRef<PlayerHandle>(null);
  const [playhead, setPlayhead] = useState(0);
  const [collapsed, setCollapsedState] = useState(storedCollapsed);
  const video = session.current;
  const setCollapsed = (v: boolean) => {
    setCollapsedState(v);
    try {
      window.localStorage.setItem(COLLAPSED_KEY, v ? "1" : "0");
    } catch {
      // not persisted
    }
  };

  const config = editor.state.pipeline.config;
  const source = config[SOURCE_KEY];
  const sourceRecord = source && typeof source === "object" && !Array.isArray(source) ? (source as Record<string, unknown>) : {};
  const pipelineInput = typeof sourceRecord.video === "string" ? sourceRecord.video : sourceRecord.mcap;
  const mcap = video !== null && isMcap(video);
  const topic = process.topic;
  const timing = video ? inputTiming(video, topic) : null;
  const segmentPlan = process.plan(false);
  const entirePlan = process.plan(true);

  const banner = session.error ? (
    <div className="deck__error" role="alert">
      <Icon name="alert" size={15} />
      <span>{session.error}</span>
      <button type="button" className="icon-btn icon-btn--xs" aria-label="Dismiss" onClick={session.dismissError}>
        <Icon name="x" />
      </button>
    </div>
  ) : null;

  const uploading = session.upload ? (
    <div className="deck__upload" role="status" aria-label="Upload">
      <div className="deck__upload-head">
        <span className="spinner" aria-hidden="true" />
        <strong>{session.upload.name}</strong>
        <span className="muted">
          {session.upload.phase === "uploading"
            ? `Uploading ${formatBytes(session.upload.loaded)} of ${formatBytes(session.upload.total)}`
            : isMcapName(session.upload.name)
              ? "Reading the topics…"
              : "Checking the video…"}
        </span>
        {session.upload.phase === "uploading" ? (
          <button type="button" className="btn btn--xs btn--ghost" onClick={session.cancelUpload}>
            Cancel
          </button>
        ) : null}
      </div>
      <div className={`progress${session.upload.phase === "processing" ? " progress--indeterminate" : ""}`}>
        <span style={{ width: `${session.upload.total ? (session.upload.loaded / session.upload.total) * 100 : 0}%` }} />
      </div>
    </div>
  ) : null;

  const actions = (compact: boolean) => (
    <div className={`deck__actions${compact ? " deck__actions--compact" : ""}`}>
      <button
        type="button"
        className="btn btn--primary"
        disabled={Boolean(process.blocked) || process.busy}
        title={process.blocked ?? (segmentPlan ? `${segmentPlan.frames} frames, ${formatDuration(segmentPlan.seconds)}` : undefined)}
        onClick={() => void process.process(false)}
      >
        {process.busy ? <span className="spinner spinner--ink" aria-hidden="true" /> : <Icon name="play" size={14} />} Process
      </button>
      {!compact ? (
        <button
          type="button"
          className="btn"
          disabled={Boolean(process.blocked) || process.busy}
          title={process.blocked ?? (entirePlan ? `${entirePlan.frames} frames, ${formatDuration(entirePlan.seconds)}` : undefined)}
          onClick={() => void process.process(true)}
        >
          {mcap ? "Process entire topic" : "Process entire video"}
          {entirePlan ? <span className="btn__aside">{formatDuration(entirePlan.seconds)}</span> : null}
        </button>
      ) : null}
      {process.blocked && !compact ? <span className="deck__blocked">{process.blocked}</span> : null}
    </div>
  );

  if (collapsed) {
    return (
      <section className="deck deck--collapsed" aria-label="Video and processing">
        {video ? <img className="deck__thumb" src={video.urls.thumb} alt="" /> : <Icon name="film" size={18} />}
        <div className="deck__titles">
          <strong>{video ? video.name : "No video"}</strong>
          {video ? (
            <span className="muted">
              {formatTime(session.segment.start)} – {formatTime(session.segment.start + session.segment.length)} · stride {session.stride}
            </span>
          ) : null}
        </div>
        {jobs.jobs.some((j) => j.status === "running" || j.status === "queued") ? <span className="chip chip--ok">Processing</span> : null}
        <div className="topbar__spacer" />
        {video ? actions(true) : null}
        <button type="button" className="icon-btn" aria-label="Expand the video panel" aria-expanded="false" onClick={() => setCollapsed(false)}>
          <Icon name="chevron-down" />
        </button>
      </section>
    );
  }

  return (
    <section className="deck" aria-label="Video and processing">
      <button type="button" className="icon-btn deck__collapse" aria-label="Collapse the video panel" aria-expanded="true" onClick={() => setCollapsed(true)}>
        <Icon name="chevron-up" />
      </button>
      {video ? (
        <>
          <div className="deck__media">
            {mcap ? (
              <McapStill video={video} topic={topic} />
            ) : (
              <Player
                ref={player}
                video={video}
                loopRange={{ start: session.segment.start, end: session.segment.start + session.segment.length }}
                onTime={setPlayhead}
              />
            )}
          </div>
          <div className="deck__controls">
            {banner}
            {uploading}
            <VideoHeader video={video} topic={topic} onChange={() => session.select(null)} />
            {mcap ? (
              <TopicPicker
                topics={topicsOf(video)}
                value={sourceTopic(config)}
                onChange={(name) => editor.updateConfig((c) => bindMcap(c, video, name))}
                useCameraInfo={usesCameraInfo(config)}
                onUseCameraInfo={(on) => editor.updateConfig((c) => setUseCameraInfo(c, on))}
              />
            ) : null}
            <SegmentBar
              duration={timing?.durationS ?? 0}
              fps={timing?.fps ?? 0}
              count={timing?.count ?? 0}
              segment={session.segment}
              stride={session.stride}
              playhead={playhead}
              onChange={(seg) => session.setSegment(seg, timing?.durationS ?? 0)}
              onStride={session.setStride}
              onSeek={(t) => {
                player.current?.seek(t);
                setPlayhead(t);
              }}
            />
            {actions(false)}
          </div>
        </>
      ) : (
        <div className="deck__intake">
          {banner}
          {uploading}
          <DropZone
            videos={session.videos}
            roots={session.roots}
            cwd={session.cwd}
            suggestedPath={typeof pipelineInput === "string" ? pipelineInput : null}
            busy={Boolean(session.upload)}
            onFiles={onFiles}
            onImport={(p) => void session.importPath(p)}
            onSelect={session.select}
          />
        </div>
      )}
      <aside className="deck__jobs" aria-label="Jobs panel">
        <h3>Jobs</h3>
        <JobPanel jobs={jobs.jobs} previewUrl={previewUrl} onCancel={(id) => void jobs.cancel(id)} onOpenRun={onOpenRun} onDismiss={jobs.dismiss} />
      </aside>
    </section>
  );
}

/** An MCAP has no preview video: its first frame, with the selected topic. */
function McapStill({ video, topic }: { video: VideoRecord; topic: McapTopic | null }) {
  return (
    <figure className="player mcap-still">
      <img className="player__video" src={video.urls.thumb} alt={`Thumbnail of ${video.name}`} />
      <figcaption className="mcap-still__caption">
        <span className="player__badge mcap-still__badge">MCAP</span>
        {topic ? (
          <>
            <code>{topic.name}</code>
            <span>
              {formatResolution(topic.width, topic.height)} · {formatRate(topic.rateHz, topic.count)}
            </span>
          </>
        ) : (
          <span>Pick an image topic</span>
        )}
      </figcaption>
    </figure>
  );
}

function VideoHeader({ video, topic, onChange }: { video: VideoRecord; topic: McapTopic | null; onChange: () => void }) {
  const mcap = isMcap(video);
  const fps = !(video.fps > 0) ? "0" : Number.isInteger(video.fps) ? String(video.fps) : video.fps.toFixed(2);
  const details = mcap
    ? topic
      ? `${formatResolution(topic.width, topic.height)} · ${formatRate(topic.rateHz, topic.count)} · ${formatTime(topic.durationS)}`
      : `${topicsOf(video).length} image topics`
    : `${video.width}×${video.height} · ${fps} fps · ${formatTime(video.duration_s)}`;
  return (
    <header className="deck__head">
      <div className="deck__titles">
        <h2 title={video.original}>
          {video.name}
          {video.missing ? (
            <span className="chip chip--bad deck__missing" title="The original is gone or changed on the server; import or upload it again">
              <Icon name="alert" size={12} /> File missing
            </span>
          ) : null}
        </h2>
        <p>
          {mcap ? "MCAP · " : ""}
          {details} · {formatBytes(video.size_bytes)}
          {!mcap && video.preview_status === "pending" ? <span className="deck__pending"> · preparing preview</span> : null}
        </p>
      </div>
      <button type="button" className="btn btn--sm" onClick={onChange}>
        <Icon name="film" size={14} /> {mcap ? "Change input" : "Change video"}
      </button>
    </header>
  );
}
