// Test catalog: same shape as GET /api/catalog. Names are test data only;
// the UI code never refers to them.
import type { ApiClient } from "../api/client";
import type { ModelKey, ModelStatus, ModelsApi, PrepareEvent, PrepareJob } from "../api/models";
import type { Catalog, ConfigDict, Field, McapTopics, ParsedConfig, ValidationError } from "../api/types";
import type { VideoApi, VideoRecord } from "../api/video";

export const thresholdField: Field = {
  name: "threshold",
  type: "float",
  nullable: false,
  default: 0.5,
  required: false,
  hint: { label: "Score threshold", min: 0, max: 1, step: 0.05 },
};

export function makeCatalog(overrides: Partial<Catalog> = {}): Catalog {
  return {
    catalog_version: 1,
    stages: [
      { yaml_key: "source", title: "Source", kind: null, multiple: false, optional: false, order: 10, general: false, has_section: true },
      { yaml_key: "detectors", title: "Detectors", kind: "detector", multiple: true, optional: false, order: 20, general: false, has_section: false },
      { yaml_key: "aggregation", title: "Aggregation", kind: "aggregator", multiple: false, optional: false, order: 30, general: false, has_section: false },
      { yaml_key: "smoothing", title: "Smoothing", kind: null, multiple: false, optional: true, order: 70, general: false, has_section: true },
      { yaml_key: "render", title: "Render", kind: null, multiple: false, optional: false, order: 140, general: true, has_section: true },
    ],
    modules: {
      detector: {
        alpha: {
          doc: "A detector.",
          schema: {
            fields: [
              thresholdField,
              { name: "variant", type: "enum", nullable: false, choices: ["small", "large"], default: "large", required: false, hint: {} },
              { name: "weights", type: "path", nullable: true, default: null, required: false, hint: {} },
            ],
          },
        },
        beta: { doc: "Another detector.", schema: { fields: [{ ...thresholdField, default: 0.25 }] } },
      },
      aggregator: {
        merge: { doc: "Merge.", schema: { fields: [{ name: "iou_thr", type: "float", nullable: false, default: 0.55, required: false, hint: {} }] } },
      },
    },
    sections: {
      source: { schema: { fields: [{ name: "video", type: "path", nullable: false, default: null, required: true, hint: {} }, { name: "stride", type: "int", nullable: false, default: 1, required: false, hint: {} }] } },
      smoothing: {
        schema: {
          fields: [
            { name: "accel_std", type: "float", nullable: false, default: 2.0, required: false, hint: { label: "Acceleration noise", unit: "m/s²", min: 0.1, max: 10, step: 0.1 } },
            { name: "tau_s", type: "float", nullable: false, default: 0.1, required: false, hint: { label: "EMA time constant", advanced: true } },
          ],
        },
      },
      render: { schema: { fields: [{ name: "hud", type: "bool", nullable: false, default: true, required: false, hint: {} }] } },
    },
    ...overrides,
  };
}

export const sampleConfig: ParsedConfig = {
  config: {
    source: { video: "a.mp4", stride: 1 },
    detectors: [{ type: "alpha", threshold: 0.5, variant: "large", weights: null }],
    aggregation: { type: "merge" },
    smoothing: { accel_std: 2.0, tau_s: 0.1 },
    render: { hud: true },
  },
  extras: {},
};

export interface FakeClient extends ApiClient {
  validateCalls: unknown[];
  errors: ValidationError[];
}

export function fakeClient(
  catalog: Catalog = makeCatalog(),
  parsed: ParsedConfig = sampleConfig,
): FakeClient {
  const client: FakeClient = {
    validateCalls: [],
    errors: [],
    catalog: async () => catalog,
    validate: async (config) => {
      client.validateCalls.push(config);
      return client.errors;
    },
    parse: async () => structuredClone(parsed),
    dump: async (config, extras) => JSON.stringify({ ...config, ...extras }),
  };
  return client;
}

/** A status row (defaults: the sample's first detector, missing 8 MB of weights). */
export function modelRow(patch: Partial<ModelStatus> = {}): ModelStatus {
  return {
    stage: "detectors",
    index: 0,
    type: "alpha",
    name: "alpha",
    state: "missing",
    layers: { assets: "missing", code: "ready", packages: "ready" },
    missing: ["weights"],
    size_bytes: 8 * 1024 * 1024,
    message: "weights: <weights root>/alpha/large.bin not found",
    ...patch,
  };
}

export interface FakeModelsApi extends ModelsApi {
  rows: ModelStatus[];
  statusCalls: ConfigDict[];
  prepared: { models: ModelKey[] | "all"; config: ConfigDict; packages?: boolean }[];
  /** `restart_required` of the next status answers. */
  restartRequired: boolean;
  cancelled: string[];
  /** Make the next status calls fail with this error. */
  failWith: Error | null;
  /** What `GET /api/models/prepare` lists (jobs started before a reload). */
  listed: PrepareJob[];
  emit(jobId: string, event: PrepareEvent): void;
}

export function fakeModelsApi(rows: ModelStatus[] = []): FakeModelsApi {
  const listeners = new Map<string, (e: PrepareEvent) => void>();
  const api: FakeModelsApi = {
    rows,
    statusCalls: [],
    prepared: [],
    cancelled: [],
    failWith: null,
    restartRequired: false,
    listed: [],
    emit: (id, event) => listeners.get(id)?.(event),
    status: async (config) => {
      api.statusCalls.push(config);
      if (api.failWith) throw api.failWith;
      const models = structuredClone(api.rows);
      return { models, total_size_bytes: models.filter((m) => m.state !== "ready").reduce((s, m) => s + (m.size_bytes ?? 0), 0), restart_required: api.restartRequired };
    },
    prepare: async (models, config, options) => {
      api.prepared.push(options?.packages ? { models, config, packages: true } : { models, config });
      const id = `prep${api.prepared.length}`;
      const chosen = models === "all" ? api.rows : api.rows.filter((r) => models.some((m) => m.stage === r.stage && m.index === r.index));
      return {
        id,
        kind: options?.packages ? "packages" : "weights",
        status: "queued",
        position: 0,
        error: null,
        created: "2026-10-31T10:00:00",
        cancel_requested: false,
        models: chosen.map((r) => ({ ...r, state: "queued", item: null, done: 0, total: null })),
        skipped: [],
      };
    },
    getPrepare: async () => Promise.reject(new Error("not used")),
    listPrepares: async () => api.listed,
    cancelPrepare: async (id) => {
      api.cancelled.push(id);
      return { id, status: "cancelled", position: null, error: "cancelled", created: "", cancel_requested: true, models: [], skipped: [] };
    },
    prepareEvents: (id, onEvent) => {
      listeners.set(id, onEvent);
      return () => listeners.delete(id);
    },
  };
  return api;
}

// ---------- MCAP inputs (GET /api/videos/{id}/topics, already camelCase) ----------

/** The image topics of the test MCAP (Task 9's fixture plus an unsupported depth topic). */
export const mcapTopics: McapTopics = {
  durationS: 1,
  cameraInfoTopics: ["/cam/camera_info", "/rect/camera_info"],
  images: [
    { name: "/cam/depth/image_raw", schema: "sensor_msgs/msg/Image", width: 16, height: 12, encoding: "16UC1", rateHz: 10, count: 2, durationS: 0.1, cameraInfoTopic: null, hasCameraInfo: false, hasDistortion: false, cameraInfoError: null, supported: false, unsupportedReason: "unsupported image encoding '16UC1'; supported: rgb8, bgr8, rgba8, bgra8, mono8, jpeg, png" },
    { name: "/cam/image_raw", schema: "sensor_msgs/msg/Image", width: 16, height: 12, encoding: "rgb8", rateHz: 2, count: 3, durationS: 1, cameraInfoTopic: "/cam/camera_info", hasCameraInfo: true, hasDistortion: true, cameraInfoError: null, supported: true, unsupportedReason: null },
    { name: "/front/image_raw/compressed", schema: "sensor_msgs/msg/CompressedImage", width: 12, height: 8, encoding: "jpeg", rateHz: 4, count: 2, durationS: 0.25, cameraInfoTopic: null, hasCameraInfo: false, hasDistortion: false, cameraInfoError: null, supported: true, unsupportedReason: null },
    { name: "/rect/image_raw", schema: "sensor_msgs/msg/Image", width: 16, height: 12, encoding: "bgr8", rateHz: 10, count: 2, durationS: 0.1, cameraInfoTopic: "/rect/camera_info", hasCameraInfo: true, hasDistortion: false, cameraInfoError: null, supported: true, unsupportedReason: null },
  ],
};

export const videoRecord: VideoRecord = {
  id: "0123456789ab",
  name: "road.mkv",
  width: 1296,
  height: 972,
  fps: 25,
  duration_s: 10,
  size_bytes: 87_000_000,
  original: "/srv/library/0123456789ab/original.mkv",
  preview: null,
  thumb: "/srv/library/0123456789ab/thumb.jpg",
  preview_status: "ready",
  created: "2026-10-31T10:00:00",
  origin: "upload",
  kind: "video",
  mcap: null,
  urls: { file: "/api/videos/0123456789ab/file", preview: "/api/videos/0123456789ab/preview", thumb: "/api/videos/0123456789ab/thumb" },
};

export const mcapRecord: VideoRecord = {
  id: "fedcba987654",
  name: "drive.mcap",
  width: 16,
  height: 12,
  fps: 2,
  duration_s: 1,
  size_bytes: 12_345,
  original: "/srv/library/fedcba987654/original.mcap",
  preview: null,
  thumb: "/srv/library/fedcba987654/thumb.jpg",
  preview_status: "ready",
  created: "2026-11-04T10:00:00",
  origin: "upload",
  kind: "mcap",
  mcap: mcapTopics,
  urls: {
    file: "/api/videos/fedcba987654/file",
    preview: "/api/videos/fedcba987654/preview",
    thumb: "/api/videos/fedcba987654/thumb",
    topics: "/api/videos/fedcba987654/topics",
  },
};

/** The test catalog plus the MCAP-related sections: source.mcap/topic, the Rectification stage and the camera. */
export function mcapCatalog(): Catalog {
  const base = makeCatalog();
  const field = (name: string, type: string, def: unknown, nullable = true): Field => ({ name, type, nullable, default: def, required: false, hint: {} });
  return makeCatalog({
    stages: [
      ...base.stages.slice(0, 1),
      { yaml_key: "rectify", title: "Rectification", kind: null, multiple: false, optional: true, order: 15, general: false, has_section: true },
      ...base.stages.slice(1),
      { yaml_key: "camera", title: "Camera", kind: null, multiple: false, optional: true, order: 120, general: true, has_section: true },
    ],
    sections: {
      ...base.sections,
      source: { schema: { fields: [field("video", "path", null), field("mcap", "path", null), field("topic", "str", null), { ...field("stride", "int", 1, false) }] } },
      rectify: { schema: { fields: [field("enabled", "bool", false, false), field("alpha", "float", 0, false)] } },
      camera: { schema: { fields: [field("focal_px", "float", null), field("calibration", "any", null)] } },
    },
  });
}

export interface FakeVideoApi extends VideoApi {
  submitted: { config: ConfigDict; videoId?: string }[];
  uploaded: File[];
  records: VideoRecord[];
}

/** A video API over `records` (upload answers the record named like the file, else the first). */
export function fakeVideoApi(records: VideoRecord[] = [videoRecord, mcapRecord]): FakeVideoApi {
  const byName = (name: string) => api.records.find((r) => r.name === name) ?? api.records[0];
  const api: FakeVideoApi = {
    submitted: [],
    uploaded: [],
    records,
    listVideos: async () => api.records,
    getVideo: async (id) => api.records.find((r) => r.id === id) ?? Promise.reject(new Error("unknown")),
    fetchTopics: async (id) => api.records.find((r) => r.id === id)?.mcap ?? Promise.reject(new Error("not an MCAP")),
    importRoots: async () => ({ roots: ["/srv/data"], cwd: "/srv" }),
    upload: async (file) => {
      api.uploaded.push(file);
      return byName(file.name);
    },
    importPath: async (path) => byName(path.split("/").pop() ?? path),
    gpuFreeMiB: async () => 20000,
    submitRun: async (config, videoId) => {
      api.submitted.push({ config, videoId });
      return `job${api.submitted.length}`;
    },
    getJob: async (id) => ({ id, status: "queued", position: 0, error: null, run_id: null, frames: 0, log_tail: [] }),
    listJobs: async () => [],
    cancelJob: async (id) => ({ id, status: "cancelled", position: null, error: null, run_id: null, frames: 0, log_tail: [] }),
    jobEvents: () => () => undefined,
    listRuns: async () => [],
    getRun: async (id) => ({ id, created: "", video: null, detectors: [], depth: null, frames: null, wall_s: null, status: "done", complete: true, has_config: true, error: null }),
    runFileUrl: (runId, name) => `/api/runs/${runId}/${name}`,
  };
  return api;
}
