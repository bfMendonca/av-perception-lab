import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { App } from "../App";
import type { ConfigDict } from "../api/types";
import type { JobEvent, JobInfo, VideoApi, VideoRecord } from "../api/video";
import { createEditorStore } from "../editor/yamlState";
import { fakeClient } from "../test/fixtures";
import { classifyFiles } from "./DropZone";

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
    importRoots: async () => ["/srv/data"],
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
  it("Process builds config with source.video, start, duration, stride and unique output.tag", async () => {
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
    const tag = (c: ConfigDict) => (c.output as { tag: string }).tag;
    expect(tag(first.config)).toBeTruthy();
    expect(tag(first.config)).not.toBe(tag(second.config));
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

  it("classifies drops into video, meta and YAML", () => {
    const f = (name: string, type = "") => new File(["x"], name, { type });
    expect(classifyFiles([f("a.h264"), f("a.meta.json", "application/json")])).toEqual({ video: expect.any(File), meta: expect.any(File) });
    expect(classifyFiles([f("p.yaml")]).yaml?.name).toBe("p.yaml");
    expect(classifyFiles([f("clip", "video/mp4")]).video?.name).toBe("clip");
    expect(classifyFiles([f("notes.txt")])).toEqual({});
  });
});
