import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { App } from "../App";
import type { ConfigDict } from "../api/types";
import type { JobEvent, JobInfo, VideoApi, VideoRecord } from "../api/video";
import { ModelsNotReady } from "../api/video";
import { createEditorStore } from "../editor/yamlState";
import { fakeClient, fakeModelsApi, modelRow, sampleConfig, type FakeModelsApi } from "../test/fixtures";
import { classifyFiles, importable } from "./DropZone";

const record: VideoRecord = {
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
  preview_status: "pending",
  created: "2026-10-31T10:00:00",
  origin: "upload",
  urls: { file: "/api/videos/0123456789ab/file", preview: "/api/videos/0123456789ab/preview", thumb: "/api/videos/0123456789ab/thumb" },
};

interface FakeVideoApi extends VideoApi {
  submitted: { config: ConfigDict; videoId?: string }[];
  free: number | null;
  emit(jobId: string, event: JobEvent): void;
}

function fakeVideoApi(): FakeVideoApi {
  const listeners = new Map<string, (e: JobEvent) => void>();
  const jobs = new Map<string, JobInfo>();
  const api: FakeVideoApi = {
    submitted: [],
    free: 20000,
    emit: (id, event) => listeners.get(id)?.(event),
    listVideos: async () => [record],
    getVideo: async () => record,
    fetchTopics: async () => Promise.reject(new Error("not an MCAP")),
    importRoots: async () => ({ roots: ["/srv/data"], cwd: "/srv" }),
    upload: async () => record,
    importPath: async () => record,
    gpuFreeMiB: async () => api.free,
    submitRun: async (config, videoId) => {
      api.submitted.push({ config, videoId });
      const id = `job${api.submitted.length}`;
      jobs.set(id, { id, status: "queued", position: api.submitted.length - 1, error: null, run_id: null, frames: 0, log_tail: [] });
      return id;
    },
    getJob: async (id) => jobs.get(id) ?? Promise.reject(new Error("unknown")),
    listJobs: async () => [],
    cancelJob: async (id) => ({ ...(jobs.get(id) as JobInfo), status: "cancelled" }),
    jobEvents: (id, onEvent) => {
      listeners.set(id, onEvent);
      return () => listeners.delete(id);
    },
    listRuns: async () => [],
    getRun: async (id) => ({ id, created: "", video: null, detectors: [], depth: null, frames: null, wall_s: null, status: "done", complete: true, has_config: true, error: null }),
    runFileUrl: (runId, name) => `/api/runs/${runId}/${name}`,
  };
  (api as unknown as { jobs: Map<string, JobInfo> }).jobs = jobs;
  return api;
}

async function renderStudio(api: FakeVideoApi) {
  const client = fakeClient();
  const store = createEditorStore(client, { debounceMs: 0 });
  await store.load("ignored", "p.yaml");
  render(<App client={client} store={store} videoApi={api} />);
  // pick the library video
  await userEvent.click(await screen.findByRole("button", { name: /road\.mkv/ }));
  const deck = screen.getByRole("region", { name: "Video and processing" });
  await waitFor(() => expect(within(deck).getByRole("button", { name: /^Process$/ })).toBeEnabled());
  return { deck, store };
}

describe("Process", () => {
  it("Process builds config with source.video, start, duration and stride (the server sets output.tag)", async () => {
    const api = fakeVideoApi();
    const { deck } = await renderStudio(api);
    const start = within(deck).getByRole("spinbutton", { name: "Start" });
    await userEvent.clear(start);
    await userEvent.type(start, "4{Enter}");
    const stride = within(deck).getByRole("spinbutton", { name: "Stride" });
    await userEvent.clear(stride);
    await userEvent.type(stride, "3{Enter}");
    expect(within(deck).getByTestId("segment-estimate")).toHaveTextContent("25 frames");
    expect(within(deck).getByTestId("segment-estimate")).toHaveTextContent("~13 s at 2 fps");

    await userEvent.click(within(deck).getByRole("button", { name: /^Process$/ }));
    await waitFor(() => expect(api.submitted).toHaveLength(1));
    // a second click queues a second job
    await userEvent.click(within(deck).getByRole("button", { name: /^Process$/ }));
    await waitFor(() => expect(api.submitted).toHaveLength(2));
    await userEvent.click(within(deck).getByRole("button", { name: /Process entire video/ }));
    await waitFor(() => expect(api.submitted).toHaveLength(3));

    const [first, second, entire] = api.submitted;
    expect(first.videoId).toBe(record.id);
    expect(first.config.source).toEqual({ video: record.original, stride: 3, start: 4, duration: 3 });
    // the rest of the editor's pipeline goes through untouched
    expect(first.config.detectors).toEqual([{ type: "alpha", threshold: 0.5, variant: "large", weights: null }]);
    expect((first.config.output as ConfigDict | undefined)?.tag).toBeUndefined();
    expect(second.config.source).toEqual(first.config.source);
    expect(entire.config.source).toEqual({ video: record.original, stride: 3, start: 0, duration: 0 });

    // the jobs panel shows the queue
    const panel = screen.getByRole("list", { name: "Jobs" });
    expect(within(panel).getAllByRole("listitem")).toHaveLength(3);
    await waitFor(() => expect(within(panel).getByText("Queued · 1 ahead")).toBeInTheDocument());
  });

  it("Process asks for confirmation when /api/gpu reports < 4000 MiB free", async () => {
    const api = fakeVideoApi();
    api.free = 3000;
    const { deck } = await renderStudio(api);
    await userEvent.click(within(deck).getByRole("button", { name: /^Process$/ }));
    const dialog = await screen.findByRole("dialog", { name: "The GPU is busy" });
    expect(dialog).toHaveTextContent("3,000 MiB");
    expect(api.submitted).toHaveLength(0);
    await userEvent.click(within(dialog).getByRole("button", { name: "Cancel" }));
    expect(api.submitted).toHaveLength(0);

    await userEvent.click(within(deck).getByRole("button", { name: /^Process$/ }));
    await userEvent.click(within(await screen.findByRole("dialog", { name: "The GPU is busy" })).getByRole("button", { name: "Process anyway" }));
    await waitFor(() => expect(api.submitted).toHaveLength(1));

    api.free = 6000; // enough: no question
    await userEvent.click(within(deck).getByRole("button", { name: /^Process$/ }));
    await waitFor(() => expect(api.submitted).toHaveLength(2));
    expect(screen.queryByRole("dialog", { name: "The GPU is busy" })).not.toBeInTheDocument();
  });

  it("job panel follows the stream: progress, failure log tail, and done jumps to the run", async () => {
    const api = fakeVideoApi();
    const { deck } = await renderStudio(api);
    await userEvent.click(within(deck).getByRole("button", { name: /^Process$/ }));
    await waitFor(() => expect(api.submitted).toHaveLength(1));
    const jobs = (api as unknown as { jobs: Map<string, JobInfo> }).jobs;
    act(() => {
      api.emit("job1", { event: "running" });
      api.emit("job1", { event: "started", run_dir: "/out/20261031_101500_road_web-x" });
      api.emit("job1", { event: "frame", n: 30 });
    });
    const bar = await screen.findByRole("progressbar", { name: "Frames processed" });
    expect(bar).toHaveAttribute("aria-valuenow", "30");
    expect(bar).toHaveAttribute("aria-valuemax", "75");
    expect(screen.getByText("30 / 75 frames")).toBeInTheDocument();

    jobs.set("job1", { id: "job1", status: "failed", position: null, error: "no frames decoded from road.mkv (is source.start past the end of the video?)", run_id: null, frames: 30, log_tail: ["Traceback", "ConfigError: no frames"] });
    act(() => api.emit("job1", { event: "end", status: "failed", error: "no frames decoded from road.mkv (is source.start past the end of the video?)" }));
    expect(await screen.findByText(/is source.start past the end/)).toBeInTheDocument();
    expect(screen.getByText(/Log tail \(2 lines\)/)).toBeInTheDocument();

    await userEvent.click(within(deck).getByRole("button", { name: /^Process$/ }));
    await waitFor(() => expect(api.submitted).toHaveLength(2));
    jobs.set("job2", { id: "job2", status: "done", position: null, error: null, run_id: "20261031_101600_road_web-y", frames: 75, log_tail: [] });
    act(() => api.emit("job2", { event: "end", status: "done", error: null }));
    await waitFor(() => expect(window.location.hash).toBe("#/runs/20261031_101600_road_web-y"));
    expect(await screen.findByRole("region", { name: "Run" })).toBeInTheDocument();
    window.location.hash = "";
  });

  it("offers the pipeline's video only when the server would import it", async () => {
    const api = fakeVideoApi();
    api.listVideos = async () => [];
    const offer = /Use the pipeline's video/;
    for (const [video, shown] of [
      ["a.mp4", false], // /srv/a.mp4: outside /srv/data
      ["data/../a.mp4", false],
      ["/elsewhere/a.mp4", false],
      ["data/s1/clip.mkv", true],
      ["/srv/data/clip.mkv", true],
    ] as const) {
      const client = fakeClient(undefined, { config: { ...sampleConfig.config, source: { video } }, extras: {} });
      const store = createEditorStore(client, { debounceMs: 0 });
      await store.load("ignored", "p.yaml");
      const { unmount } = render(<App client={client} store={store} videoApi={api} />);
      await screen.findByText(/Allowed folders/);
      if (shown) expect(await screen.findByRole("button", { name: offer })).toBeInTheDocument();
      else expect(screen.queryByRole("button", { name: offer })).not.toBeInTheDocument();
      unmount();
    }
    expect(importable("data/x", [], "/srv")).toBe(false);
    expect(importable("/srv/database/x", ["/srv/data"], "/")).toBe(false);
  });

  it("a video whose file is missing is flagged and cannot be processed", async () => {
    const api = fakeVideoApi();
    const gone = { ...record, missing: true };
    api.listVideos = async () => [gone];
    const client = fakeClient();
    const store = createEditorStore(client, { debounceMs: 0 });
    await store.load("ignored", "p.yaml");
    render(<App client={client} store={store} videoApi={api} />);
    const item = await screen.findByRole("button", { name: /road\.mkv/ });
    expect(item).toHaveTextContent("File missing");
    await userEvent.click(item);
    const deck = screen.getByRole("region", { name: "Video and processing" });
    expect(within(deck).getByText("File missing")).toBeInTheDocument();
    expect(within(deck).getByRole("button", { name: /^Process$/ })).toBeDisabled();
    expect(within(deck).getByText(/missing or changed on the server/)).toBeInTheDocument();
  });

  it("process is disabled while any model is not ready and enabled when all are ready", async () => {
    const api = fakeVideoApi();
    const models: FakeModelsApi = fakeModelsApi([modelRow(), modelRow({ stage: "depth", index: null, name: "deep", type: "beta" })]);
    const client = fakeClient();
    const store = createEditorStore(client, { debounceMs: 0 });
    await store.load("ignored", "p.yaml");
    render(<App client={client} store={store} videoApi={api} modelsApi={models} />);
    await userEvent.click(await screen.findByRole("button", { name: /road\.mkv/ }));
    const deck = screen.getByRole("region", { name: "Video and processing" });
    const button = within(deck).getByRole("button", { name: /^Process$/ });
    await waitFor(() => expect(button).toBeDisabled());
    expect(button).toHaveAttribute("title", "2 models are not ready — Prepare them first.");
    expect(within(deck).getByText("2 models are not ready — Prepare them first.")).toBeInTheDocument();
    expect(within(deck).getByRole("button", { name: /Process entire video/ })).toBeDisabled();

    // all ready (e.g. after a Prepare elsewhere): Process unlocks
    models.rows = [modelRow({ state: "ready", missing: [], message: null }), modelRow({ stage: "depth", index: null, name: "deep", type: "beta", state: "ready", missing: [], message: null })];
    store.setOption(["detectors", 0, "threshold"], 0.6); // any edit checks again
    await waitFor(() => expect(button).toBeEnabled());
    await userEvent.click(button);
    await waitFor(() => expect(api.submitted).toHaveLength(1));
  });

  it("409 from submit shows the list of models", async () => {
    const api = fakeVideoApi();
    const models = fakeModelsApi([modelRow({ state: "ready", missing: [], message: null })]);
    api.submitRun = async () => {
      throw new ModelsNotReady([modelRow({ message: "weights: <weights root>/alpha/large.bin not found" }), modelRow({ stage: "camera.estimate", index: null, name: "focal", type: "beta", state: "failed", message: "unsafe target" })]);
    };
    const client = fakeClient();
    const store = createEditorStore(client, { debounceMs: 0 });
    await store.load("ignored", "p.yaml");
    render(<App client={client} store={store} videoApi={api} modelsApi={models} />);
    await userEvent.click(await screen.findByRole("button", { name: /road\.mkv/ }));
    const deck = screen.getByRole("region", { name: "Video and processing" });
    await waitFor(() => expect(within(deck).getByRole("button", { name: /^Process$/ })).toBeEnabled());
    const calls = models.statusCalls.length;
    await userEvent.click(within(deck).getByRole("button", { name: /^Process$/ }));
    const dialog = await screen.findByRole("dialog", { name: "Models not ready" });
    const list = within(dialog).getByRole("list", { name: "Models not ready" });
    const items = within(list).getAllByRole("listitem");
    expect(items).toHaveLength(2);
    expect(items[0]).toHaveTextContent("detectors[0]");
    expect(items[0]).toHaveTextContent("alpha");
    expect(items[0]).toHaveTextContent("<weights root>/alpha/large.bin not found");
    expect(items[1]).toHaveTextContent("camera.estimate");
    expect(items[1]).toHaveTextContent("Failed");
    await waitFor(() => expect(models.statusCalls.length).toBeGreaterThan(calls)); // checked again
    await userEvent.click(within(dialog).getByRole("button", { name: "Prepare all" }));
    await waitFor(() => expect(models.prepared).toHaveLength(1));
    expect(screen.queryByRole("dialog", { name: "Models not ready" })).not.toBeInTheDocument();
  });

  it("classifies drops into video, meta and YAML", () => {
    const f = (name: string, type = "") => new File(["x"], name, { type });
    expect(classifyFiles([f("a.h264"), f("a.meta.json", "application/json")])).toEqual({ video: expect.any(File), meta: expect.any(File) });
    expect(classifyFiles([f("p.yaml")]).yaml?.name).toBe("p.yaml");
    expect(classifyFiles([f("clip", "video/mp4")]).video?.name).toBe("clip");
    expect(classifyFiles([f("notes.txt")])).toEqual({});
  });
});
