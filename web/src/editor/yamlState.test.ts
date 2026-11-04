import { describe, expect, it, vi } from "vitest";
import { fakeClient, makeCatalog, sampleConfig } from "../test/fixtures";
import { createEditorStore } from "./yamlState";

function loadedStore() {
  const store = createEditorStore(fakeClient(), { debounceMs: 0 });
  store.setCatalog(makeCatalog());
  store.loadParsed(structuredClone(sampleConfig), "sample.yaml");
  return store;
}

describe("editor state", () => {
  it("addModule appends a detector spec with schema defaults and marks dirty", () => {
    const store = loadedStore();
    expect(store.getState().dirty).toBe(false);
    const result = store.addModule("detectors", "beta");
    expect(result.ok).toBe(true);
    const detectors = store.toConfig().detectors as unknown[];
    expect(detectors).toHaveLength(2);
    expect(detectors[1]).toEqual({ type: "beta", threshold: 0.25 });
    expect(store.getState().dirty).toBe(true);
  });

  it("removeModule on a single-slot stage is rejected", () => {
    const store = loadedStore();
    const before = store.toConfig();
    const result = store.removeModule("aggregation", 0);
    expect(result.ok).toBe(false);
    expect(store.toConfig()).toEqual(before);
    expect(store.getState().dirty).toBe(false);
    // a multi-slot stage accepts it
    store.addModule("detectors", "beta");
    expect(store.removeModule("detectors", 0).ok).toBe(true);
    expect((store.toConfig().detectors as { type: string }[]).map((d) => d.type)).toEqual(["beta"]);
  });

  it("setOption updates nested path and diff() lists before/after", () => {
    const store = loadedStore();
    store.setOption(["smoothing", "accel_std"], 3.5);
    store.setOption(["detectors", 0, "threshold"], 0.6);
    const config = store.toConfig() as { smoothing: { accel_std: number } };
    expect(config.smoothing.accel_std).toBe(3.5);
    expect(store.diff()).toEqual([
      { path: "detectors[0].threshold", before: 0.5, after: 0.6 },
      { path: "smoothing.accel_std", before: 2.0, after: 3.5 },
    ]);
    store.setOption(["smoothing", "accel_std"], 2.0);
    store.setOption(["detectors", 0, "threshold"], 0.5);
    expect(store.diff()).toEqual([]);
    expect(store.getState().dirty).toBe(false);
  });

  it("load keeps unknown keys in extras and toConfig re-emits them", async () => {
    const parsed = { config: { detectors: [{ type: "alpha" }] }, extras: { my_notes: { a: 1 } } };
    const store = createEditorStore(fakeClient(makeCatalog(), parsed), { debounceMs: 0 });
    await store.load("detectors: [{type: alpha}]\nmy_notes: {a: 1}\n", "notes.yaml");
    expect(store.getState().pipeline.extras).toEqual({ my_notes: { a: 1 } });
    expect(store.getState().pipeline.config).not.toHaveProperty("my_notes");
    expect(store.toConfig()).toEqual({ detectors: [{ type: "alpha" }], my_notes: { a: 1 } });
    expect(store.getState().fileName).toBe("notes.yaml");
  });
});

describe("validation", () => {
  it("an edit made while a validation request is in flight discards its stale response", async () => {
    const client = fakeClient();
    const pending: ((errors: { path: string; message: string }[]) => void)[] = [];
    client.validate = (config) => {
      client.validateCalls.push(config);
      return new Promise((resolve) => pending.push(resolve));
    };
    const store = createEditorStore(client, { debounceMs: 0 });
    store.setCatalog(makeCatalog());
    store.loadParsed(structuredClone(sampleConfig), "s.yaml");
    await vi.waitFor(() => expect(pending).toHaveLength(1)); // first request in flight
    store.setOption(["smoothing", "accel_std"], 99); // edit during the flight
    pending[0]([]); // the old request answers "valid"
    await Promise.resolve();
    expect(store.getState().validation).toBe("pending");
    await vi.waitFor(() => expect(pending).toHaveLength(2));
    pending[1]([{ path: "smoothing.accel_std", message: "too large" }]);
    await vi.waitFor(() => expect(store.getState().validation).toBe("invalid"));
    expect(store.getState().errors).toEqual([{ path: "smoothing.accel_std", message: "too large" }]);
  });
});

describe("loadPreset", () => {
  const preset = () => ({
    config: { source: { video: "p.mp4" }, output: { out_root: "p" }, detectors: [{ type: "beta" }] },
    extras: {},
  });

  it("keeps the current source and output", () => {
    const store = loadedStore();
    store.setOption(["source", "video"], "a.mp4");
    store.setOption(["output", "out_root"], "o");
    store.loadPreset(preset());
    const config = store.toConfig() as Record<string, unknown>;
    expect(config.source).toMatchObject({ video: "a.mp4" });
    expect(config.output).toMatchObject({ out_root: "o" });
    expect(config.detectors).toEqual([{ type: "beta" }]);
  });

  it("on a fresh editor adds no source or output", () => {
    const store = createEditorStore(fakeClient(), { debounceMs: 0 });
    store.setCatalog(makeCatalog());
    store.startEmpty();
    store.loadPreset(preset());
    const config = store.toConfig() as Record<string, unknown>;
    expect("source" in config).toBe(false);
    expect("output" in config).toBe(false);
    expect(config.detectors).toEqual([{ type: "beta" }]);
  });

  it("leaves the editor clean", () => {
    const store = loadedStore();
    store.loadPreset(preset());
    expect(store.getState().dirty).toBe(false);
    expect(store.getState().fileName).toBeNull();
    expect(store.diff()).toEqual([]);
  });
});
