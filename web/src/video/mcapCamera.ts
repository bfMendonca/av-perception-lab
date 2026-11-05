// What an MCAP input allows, and how choosing a topic, the CameraInfo
// intrinsics and the Rectification stage change the pipeline config. Pure:
// the deck, the topic picker and the Rectification lane share these rules
// (the server checks them again on submit).
import type { ConfigDict, McapTopic } from "../api/types";
import { isMcap, type VideoRecord } from "../api/video";
import { SOURCE_KEY } from "./segment";

export const RECTIFY_KEY = "rectify";
const CAMERA_KEY = "camera";

export type RectifyState = "available" | "no_camera_info" | "already_rectified" | "unsupported";

export interface Availability<S extends string> {
  state: S;
  /** Why it cannot be used (absent when available). */
  reason?: string;
}

const NO_TOPIC = "pick an image topic";
const NO_CAMERA_INFO = "no CameraInfo for this topic";

/** Whether the Rectification stage can be turned on for `topic`. */
export function rectifyAvailability(topic: McapTopic | null | undefined): Availability<RectifyState> {
  if (!topic) return { state: "unsupported", reason: NO_TOPIC };
  if (!topic.supported) return { state: "unsupported", reason: topic.unsupportedReason ?? "this topic cannot be read" };
  if (topic.cameraInfoError) return { state: "unsupported", reason: topic.cameraInfoError };
  if (!topic.hasCameraInfo) return { state: "no_camera_info", reason: NO_CAMERA_INFO };
  if (!topic.hasDistortion) return { state: "already_rectified", reason: "already rectified" };
  return { state: "available" };
}

/** Whether "use the CameraInfo" (`camera.calibration.from_source`) can be ticked for `topic`. */
export function cameraInfoAvailability(topic: McapTopic | null | undefined): { available: boolean; reason?: string } {
  if (!topic) return { available: false, reason: NO_TOPIC };
  if (!topic.supported) return { available: false, reason: topic.unsupportedReason ?? "this topic cannot be read" };
  if (!topic.hasCameraInfo) return { available: false, reason: NO_CAMERA_INFO };
  if (topic.cameraInfoError) return { available: false, reason: topic.cameraInfoError };
  return { available: true };
}

const asRecord = (v: unknown): ConfigDict => (v && typeof v === "object" && !Array.isArray(v) ? (v as ConfigDict) : {});
const without = (obj: ConfigDict, ...keys: string[]) => Object.fromEntries(Object.entries(obj).filter(([k]) => !keys.includes(k)));

export const topicsOf = (record: VideoRecord | null | undefined): McapTopic[] => (isMcap(record) ? (record?.mcap?.images ?? []) : []);
export const findTopic = (record: VideoRecord | null | undefined, name: unknown): McapTopic | null =>
  typeof name === "string" ? (topicsOf(record).find((t) => t.name === name) ?? null) : null;
export const firstSupported = (record: VideoRecord | null | undefined): McapTopic | null => topicsOf(record).find((t) => t.supported) ?? null;

/** `source.topic` of a config (null when unset). */
export function sourceTopic(config: ConfigDict): string | null {
  const topic = asRecord(config[SOURCE_KEY]).topic;
  return typeof topic === "string" && topic ? topic : null;
}

/** `camera.calibration.from_source` is on. */
export function usesCameraInfo(config: ConfigDict): boolean {
  return asRecord(asRecord(config[CAMERA_KEY]).calibration).from_source === true;
}

/** `rectify.enabled` is on. */
export function rectifyEnabled(config: ConfigDict): boolean {
  return asRecord(config[RECTIFY_KEY]).enabled === true;
}

/** The config does not read this MCAP yet (another file, no topic, or a `source.video`). */
export function needsBinding(config: ConfigDict, record: VideoRecord | null | undefined): boolean {
  if (!record || !isMcap(record)) return false;
  const source = asRecord(config[SOURCE_KEY]);
  return source.mcap !== record.original || !sourceTopic(config) || "video" in source;
}

/**
 * Read `record` (an MCAP) from `topicName`, or else the config's topic when the
 * record has it supported, or else its first supported topic. `source.video`
 * leaves; the CameraInfo and Rectification choices the topic cannot honour
 * are turned off.
 */
export function bindMcap(config: ConfigDict, record: VideoRecord, topicName?: string): ConfigDict {
  const current = findTopic(record, sourceTopic(config));
  const topic = topicName !== undefined ? findTopic(record, topicName) : current?.supported ? current : firstSupported(record);
  if (!topic) return config;
  const source = { ...without(asRecord(config[SOURCE_KEY]), "video"), mcap: record.original, topic: topic.name };
  let next: ConfigDict = { ...config, [SOURCE_KEY]: source };
  if (usesCameraInfo(next) && !cameraInfoAvailability(topic).available) next = setUseCameraInfo(next, false);
  if (rectifyEnabled(next) && rectifyAvailability(topic).state !== "available") next = setRectify(next, false);
  return next;
}

/** Back to a video: the MCAP keys and the choices that need one leave the config. */
export function unbindMcap(config: ConfigDict): ConfigDict {
  let next = config;
  const source = asRecord(config[SOURCE_KEY]);
  if ("mcap" in source || "topic" in source) next = { ...next, [SOURCE_KEY]: without(source, "mcap", "topic") };
  if (usesCameraInfo(next)) next = setUseCameraInfo(next, false);
  if (rectifyEnabled(next)) next = setRectify(next, false);
  return next;
}

/**
 * Tick or untick "use the CameraInfo". On, the calibration becomes
 * `{from_source: true}` (keeping `cross_check`/`check_lens`) and `focal_px`
 * leaves: both are exclusive with it. Off removes a from_source calibration.
 */
export function setUseCameraInfo(config: ConfigDict, on: boolean): ConfigDict {
  if (!(CAMERA_KEY in config) && !on) return config;
  const camera = asRecord(config[CAMERA_KEY]);
  const calibration = asRecord(camera.calibration);
  if (on) {
    const kept = Object.fromEntries(Object.entries(calibration).filter(([k]) => k === "cross_check" || k === "check_lens"));
    return { ...config, [CAMERA_KEY]: { ...without(camera, "focal_px"), calibration: { from_source: true, ...kept } } };
  }
  if (calibration.from_source !== true) return config;
  return { ...config, [CAMERA_KEY]: without(camera, "calibration") };
}

/**
 * Turn the Rectification stage on (from `defaults`, the section's defaults) or
 * off (its key leaves). On removes `camera.focal_px` and a calibration file:
 * rectify takes K from the CameraInfo (a from_source calibration may stay).
 */
export function setRectify(config: ConfigDict, on: boolean, defaults: ConfigDict = {}): ConfigDict {
  if (!on) return RECTIFY_KEY in config ? without(config, RECTIFY_KEY) : config;
  const next: ConfigDict = { ...config, [RECTIFY_KEY]: { ...defaults, ...asRecord(config[RECTIFY_KEY]), enabled: true } };
  if (!(CAMERA_KEY in config)) return next;
  const camera = asRecord(config[CAMERA_KEY]);
  const calibration = asRecord(camera.calibration);
  const drop = ["focal_px", ...("calibration" in camera && calibration.from_source !== true ? ["calibration"] : [])];
  return { ...next, [CAMERA_KEY]: without(camera, ...drop) };
}

/** "16×12", or "size unknown" for a topic whose frames did not decode. */
export function formatResolution(width: number, height: number): string {
  return width > 0 && height > 0 ? `${width}×${height}` : "size unknown";
}

/** "2 Hz", "12.5 Hz"; "1 message" when there is no rate (fewer than two messages). */
export function formatRate(hz: number, count: number): string {
  if (!(hz > 0)) return count === 1 ? "1 message" : `${count} messages`;
  return `${hz >= 10 ? Math.round(hz) : Math.round(hz * 10) / 10} Hz`;
}

export interface InputTiming {
  /** Frames per second (0: unknown, a single message). */
  fps: number;
  durationS: number;
  /** Frames in the input (0: unknown, a video). */
  count: number;
}

/** Rate and length of what a run reads: the selected topic of an MCAP, else the video. */
export function inputTiming(record: VideoRecord, topic: McapTopic | null): InputTiming {
  if (isMcap(record) && topic) return { fps: topic.rateHz, durationS: topic.durationS, count: topic.count };
  return { fps: record.fps > 0 ? record.fps : 0, durationS: record.duration_s > 0 ? record.duration_s : 0, count: 0 };
}
