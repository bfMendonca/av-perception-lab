// Client of the video library, run queue and GPU routes
// (src/av_perception_lab/web/app.py); injected so tests can fake it.
import type { ConfigDict, McapTopic, McapTopics, ValidationError } from "./types";
import { ApiError } from "./client";
import type { ModelStatus } from "./models";

export interface VideoRecord {
  id: string;
  name: string;
  width: number;
  height: number;
  /** Mean frame rate (variable-rate videos report their average). */
  fps: number;
  duration_s: number;
  size_bytes: number;
  /** Absolute path of the original on the server (what source.video, or source.mcap, gets). */
  original: string;
  /** Path of the browser-playable MP4, null until the background encode ends. */
  preview: string | null;
  thumb: string;
  preview_status: "pending" | "ready" | "failed" | (string & {});
  created: string;
  origin: string;
  /** The original is gone, or a referenced import changed since it was added. */
  missing?: boolean;
  /** A video, or a ROS 2 MCAP (no preview video; its frames come from an image topic). Absent on older servers: a video. */
  kind?: "video" | "mcap";
  /** The probed image topics of an MCAP (null for a video). */
  mcap?: McapTopics | null;
  urls: { file: string; preview: string; thumb: string; topics?: string };
}

export const isMcap = (record: VideoRecord | null | undefined): boolean => record?.kind === "mcap";

// The server answers snake_case; the topic types are camelCase.
type RawTopic = Record<string, unknown>;
type RawTopics = { duration_s?: unknown; camera_info_topics?: unknown; images?: unknown };

const num = (v: unknown) => (typeof v === "number" && Number.isFinite(v) ? v : 0);
const strOrNull = (v: unknown) => (typeof v === "string" ? v : null);

function toTopic(t: RawTopic): McapTopic {
  return {
    name: String(t.name ?? ""),
    schema: String(t.schema ?? ""),
    width: num(t.width),
    height: num(t.height),
    encoding: String(t.encoding ?? ""),
    rateHz: num(t.rate_hz),
    count: num(t.count),
    durationS: num(t.duration_s),
    cameraInfoTopic: strOrNull(t.camera_info_topic),
    hasCameraInfo: t.has_camera_info === true,
    hasDistortion: t.has_distortion === true,
    cameraInfoError: strOrNull(t.camera_info_error),
    supported: t.supported !== false, // older records have no flag: supported
    unsupportedReason: strOrNull(t.unsupported_reason),
  };
}

export function toTopics(raw: RawTopics): McapTopics {
  return {
    durationS: num(raw.duration_s),
    cameraInfoTopics: Array.isArray(raw.camera_info_topics) ? raw.camera_info_topics.map(String) : [],
    images: Array.isArray(raw.images) ? raw.images.map((t) => toTopic(t as RawTopic)) : [],
  };
}

function toRecord(raw: VideoRecord): VideoRecord {
  const mcap = raw.mcap as unknown as RawTopics | null | undefined;
  return { ...raw, mcap: mcap ? toTopics(mcap) : null };
}

export type JobStatus = "queued" | "running" | "done" | "failed" | "cancelled" | (string & {});

export interface JobInfo {
  id: string;
  status: JobStatus;
  /** Place in the waiting line (0 = next); null once started. */
  position: number | null;
  error: string | null;
  run_id: string | null;
  frames: number;
  log_tail: string[];
  /** When it was submitted (ISO 8601); absent in older servers. */
  created?: string;
  video_id?: string | null;
  /** Name of the video it runs (the library name when it has one). */
  video?: string | null;
}

/** One server-sent event of a job (`queued`, `running`, `started`, `frame`, `done`, `error`, `end`, ...). */
export interface JobEvent {
  event: string;
  [key: string]: unknown;
}

export interface RunRecord {
  id: string;
  created: string;
  video: string | null;
  detectors: string[];
  depth: string | null;
  frames: number | null;
  wall_s: number | null;
  status: string;
  complete: boolean;
  has_config: boolean;
  error: string | null;
  /** Library video the run processed (null for CLI runs). */
  video_id?: string | null;
  /** Last log lines of a failed job without a run folder. */
  log_tail?: string[];
  [key: string]: unknown;
}

/** POST /api/runs answered 422: the config (or the video id) is invalid. */
export class RunRejected extends Error {
  constructor(readonly errors: ValidationError[]) {
    super(errors.length ? `${errors[0].path ? `${errors[0].path}: ` : ""}${errors[0].message}` : "The run was rejected.");
    this.name = "RunRejected";
  }
}

/** POST /api/runs answered 409: some models of the config are not ready (Prepare them first). */
export class ModelsNotReady extends Error {
  constructor(readonly models: ModelStatus[]) {
    super(`${models.length} model${models.length === 1 ? " is" : "s are"} not ready`);
    this.name = "ModelsNotReady";
  }
}

export interface UploadOptions {
  /** The `.meta.json` of a raw `.h264` (sensor timestamps). */
  meta?: File;
  onProgress?: (loaded: number, total: number) => void;
  signal?: AbortSignal;
}

export interface ImportRoots {
  roots: string[];
  /** The server's working directory (relative paths start there). */
  cwd: string;
}

export interface VideoApi {
  listVideos(): Promise<VideoRecord[]>;
  getVideo(id: string): Promise<VideoRecord>;
  /** The image topics of an MCAP entry (the same object as its record's `mcap`). */
  fetchTopics(videoId: string): Promise<McapTopics>;
  /** Server folders a path may be imported from, and where relative paths start. */
  importRoots(): Promise<ImportRoots>;
  upload(file: File, options?: UploadOptions): Promise<VideoRecord>;
  importPath(path: string): Promise<VideoRecord>;
  /** Free GPU memory in MiB; null when unknown. */
  gpuFreeMiB(): Promise<number | null>;
  submitRun(config: ConfigDict, videoId?: string): Promise<string>;
  getJob(id: string): Promise<JobInfo>;
  /** This server's unfinished jobs and its latest finished ones, newest first. */
  listJobs(): Promise<JobInfo[]>;
  cancelJob(id: string): Promise<JobInfo>;
  /** Subscribe to a job's events (replayed from the start; a reconnect resumes after the last one); returns the unsubscribe. */
  jobEvents(id: string, onEvent: (event: JobEvent) => void, onError?: () => void): () => void;
  listRuns(): Promise<RunRecord[]>;
  getRun(id: string): Promise<RunRecord>;
  /** URL of a run file (`video`, `preview`, `config`, `detections`). */
  runFileUrl(runId: string, name: string): string;
}

async function errorOf(response: Response): Promise<ApiError> {
  let message = `${response.status} ${response.statusText}`.trim();
  try {
    const body = (await response.json()) as { detail?: unknown };
    if (typeof body.detail === "string") message = body.detail;
  } catch {
    // keep the status line
  }
  return new ApiError(message, response.status);
}

export function createVideoApi(base = "", fetchImpl: typeof fetch = (...a) => fetch(...a)): VideoApi {
  async function call<T>(path: string, init?: RequestInit): Promise<T> {
    const response = await fetchImpl(`${base}${path}`, init);
    if (!response.ok) throw await errorOf(response);
    return (await response.json()) as T;
  }
  const post = (body: unknown): RequestInit => ({ method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const enc = encodeURIComponent;

  return {
    listVideos: async () => (await call<VideoRecord[]>("/api/videos")).map(toRecord),
    getVideo: async (id) => toRecord(await call<VideoRecord>(`/api/videos/${enc(id)}`)),
    fetchTopics: async (id) => toTopics(await call<RawTopics>(`/api/videos/${enc(id)}/topics`)),
    importRoots: () => call<ImportRoots>("/api/videos/roots"),
    importPath: async (path) => toRecord(await call<VideoRecord>("/api/videos/import", post({ path }))),
    upload: (file, options = {}) =>
      // XMLHttpRequest: fetch has no upload progress. The server streams the
      // body to disk, so a large file never sits in its memory.
      new Promise<VideoRecord>((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        const form = new FormData();
        form.append("file", file, file.name);
        if (options.meta) form.append("meta", options.meta, options.meta.name);
        xhr.open("POST", `${base}/api/videos`);
        xhr.responseType = "json";
        xhr.upload.onprogress = (e) => options.onProgress?.(e.loaded, e.lengthComputable ? e.total : file.size);
        xhr.onload = () => {
          if (xhr.status >= 200 && xhr.status < 300) resolve(toRecord(xhr.response as VideoRecord));
          else {
            const detail = (xhr.response as { detail?: unknown } | null)?.detail;
            reject(new ApiError(typeof detail === "string" ? detail : `${xhr.status} ${xhr.statusText}`, xhr.status));
          }
        };
        xhr.onerror = () => reject(new ApiError("the upload failed (network error)", 0));
        xhr.onabort = () => reject(new DOMException("upload cancelled", "AbortError"));
        options.signal?.addEventListener("abort", () => xhr.abort());
        xhr.send(form);
      }),
    gpuFreeMiB: async () => (await call<{ free_mib: number | null }>("/api/gpu")).free_mib,
    submitRun: async (config, videoId) => {
      const response = await fetchImpl(`${base}/api/runs`, post({ config, ...(videoId ? { video_id: videoId } : {}) }));
      if (response.status === 422) {
        const body = (await response.json().catch(() => ({}))) as { errors?: ValidationError[]; detail?: unknown };
        throw new RunRejected(body.errors ?? [{ path: "", message: typeof body.detail === "string" ? body.detail : "invalid request" }]);
      }
      if (response.status === 409) {
        const body = (await response.clone().json().catch(() => ({}))) as { models_not_ready?: ModelStatus[] };
        if (Array.isArray(body.models_not_ready)) throw new ModelsNotReady(body.models_not_ready);
      }
      if (!response.ok) throw await errorOf(response);
      return ((await response.json()) as { job_id: string }).job_id;
    },
    getJob: (id) => call<JobInfo>(`/api/jobs/${enc(id)}`),
    listJobs: () => call<JobInfo[]>("/api/jobs"),
    cancelJob: (id) => call<JobInfo>(`/api/jobs/${enc(id)}`, { method: "DELETE" }),
    jobEvents: (id, onEvent, onError) => {
      const source = new EventSource(`${base}/api/jobs/${enc(id)}/events`);
      let ended = false;
      source.onmessage = (e: MessageEvent<string>) => {
        try {
          const event = JSON.parse(e.data) as JobEvent;
          if (event.event === "end") {
            ended = true;
            source.close(); // otherwise EventSource reconnects and replays
          }
          onEvent(event);
        } catch {
          // a malformed line is ignored
        }
      };
      source.onerror = () => {
        if (!ended && source.readyState === EventSource.CLOSED) onError?.();
      };
      return () => source.close();
    },
    listRuns: () => call<RunRecord[]>("/api/runs"),
    getRun: (id) => call<RunRecord>(`/api/runs/${enc(id)}`),
    runFileUrl: (runId, name) => `${base}/api/runs/${enc(runId)}/${enc(name)}`,
  };
}
