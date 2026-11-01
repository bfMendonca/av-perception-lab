// Test catalog: same shape as GET /api/catalog. Names are test data only;
// the UI code never refers to them.
import type { ApiClient } from "../api/client";
import type { ModelKey, ModelStatus, ModelsApi, PrepareEvent, PrepareJob } from "../api/models";
import type { Catalog, ConfigDict, Field, ParsedConfig, ValidationError } from "../api/types";

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
  prepared: { models: ModelKey[] | "all"; config: ConfigDict }[];
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
    listed: [],
    emit: (id, event) => listeners.get(id)?.(event),
    status: async (config) => {
      api.statusCalls.push(config);
      if (api.failWith) throw api.failWith;
      const models = structuredClone(api.rows);
      return { models, total_size_bytes: models.filter((m) => m.state !== "ready").reduce((s, m) => s + (m.size_bytes ?? 0), 0) };
    },
    prepare: async (models, config) => {
      api.prepared.push({ models, config });
      const id = `prep${api.prepared.length}`;
      const chosen = models === "all" ? api.rows : api.rows.filter((r) => models.some((m) => m.stage === r.stage && m.index === r.index));
      return {
        id,
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
