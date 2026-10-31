// Segment math of the Process action: which part of the video runs, how many
// frames that is and roughly how long it takes.
import type { ConfigDict } from "../api/types";

// The run contract of `cli.run` (config.RunConfig), not a pipeline stage: the
// input section (the server forces the output section under its out_root).
export const SOURCE_KEY = "source";

/** Measured throughput of the recommended pipeline (frames per second). */
export const PROCESSING_FPS = 2;
/** Shortest segment the bar allows (seconds). */
export const MIN_SEGMENT_S = 0.1;

export interface SourceSegment {
  start: number;
  /** Seconds; 0 = to the end of the video. */
  duration: number;
  stride: number;
}

const ms = (s: number) => Math.round(s * 1000) / 1000;

/**
 * The `source.start/duration/stride` of a segment. Start is clamped inside the
 * video, the duration to what is left after it (when `videoDuration` is
 * known), the stride to an integer >= 1. A duration of 0 stays 0 ("to the end").
 */
export function segmentToSource(start: number, duration: number, stride: number, videoDuration = Infinity): SourceSegment {
  const total = Number.isFinite(videoDuration) && videoDuration > 0 ? videoDuration : Infinity;
  const s = Math.min(Math.max(Number.isFinite(start) ? start : 0, 0), Number.isFinite(total) ? Math.max(total - MIN_SEGMENT_S, 0) : Infinity);
  let d = Math.max(Number.isFinite(duration) ? duration : 0, 0);
  if (d > 0 && Number.isFinite(total)) d = Math.min(d, total - s);
  return { start: ms(s), duration: ms(d), stride: Math.max(1, Math.round(Number.isFinite(stride) ? stride : 1)) };
}

/** Frames the run processes: every `stride`-th frame of `durationS` seconds at `fps`. */
export function expectedFrames(durationS: number, fps: number, stride: number): number {
  if (!(durationS > 0) || !(fps > 0)) return 0;
  return Math.max(1, Math.ceil((durationS * fps) / Math.max(1, Math.round(stride))));
}

/** Rough wall time (s) of `frames` at the measured processing rate. */
export function estimateSeconds(frames: number, fpsProcessing = PROCESSING_FPS): number {
  return fpsProcessing > 0 ? frames / fpsProcessing : 0;
}

/** "1:05.3" (or "1:02:03.0" past an hour). */
export function formatTime(seconds: number): string {
  const t = Math.max(0, Number.isFinite(seconds) ? seconds : 0);
  const h = Math.floor(t / 3600);
  const m = Math.floor((t % 3600) / 60);
  const s = (t % 60).toFixed(1).padStart(4, "0");
  return h ? `${h}:${String(m).padStart(2, "0")}:${s}` : `${m}:${s}`;
}

/** "~38 s", "~2 min 5 s", "~1 h 4 min". */
export function formatDuration(seconds: number): string {
  const t = Math.max(0, Math.round(seconds));
  if (t < 60) return `~${t} s`;
  if (t < 3600) return `~${Math.floor(t / 60)} min${t % 60 ? ` ${t % 60} s` : ""}`;
  return `~${Math.floor(t / 3600)} h ${Math.round((t % 3600) / 60)} min`;
}

/**
 * The pipeline config of the editor with the video and the segment filled in.
 * `output` is left as it is: the server sets the run folder and its tag.
 */
export function buildRunConfig(config: ConfigDict, videoPath: string, segment: SourceSegment): ConfigDict {
  const source = config[SOURCE_KEY];
  const asRecord = (v: unknown) => (v && typeof v === "object" && !Array.isArray(v) ? (v as ConfigDict) : {});
  return {
    ...config,
    [SOURCE_KEY]: { ...asRecord(source), video: videoPath, start: segment.start, duration: segment.duration, stride: segment.stride },
  };
}
