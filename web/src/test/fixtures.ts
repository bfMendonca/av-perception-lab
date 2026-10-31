// Test catalog: same shape as GET /api/catalog. Names are test data only;
// the UI code never refers to them.
import type { ApiClient } from "../api/client";
import type { Catalog, Field, ParsedConfig, ValidationError } from "../api/types";

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
