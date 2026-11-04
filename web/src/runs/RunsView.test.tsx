import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { RunsApi } from "../api/runs";
import type { RunRecord, VideoApi } from "../api/video";
import { parseHash, routeHash } from "../ui/route";
import { ApiError } from "../api/client";
import { formatBytes } from "./format";
import { RunsView } from "./RunsView";

const failed: RunRecord = {
  id: "job-0123abcd",
  created: "2026-10-31T10:00:00",
  video: "road.mkv",
  detectors: [],
  depth: null,
  frames: null,
  wall_s: null,
  status: "failed",
  complete: false,
  has_config: true,
  error: "RuntimeError: CUDA out of memory",
  log_tail: ["loading weights", "RuntimeError: CUDA out of memory"],
};

describe("RunsView", () => {
  it("shows a failed job's error and log instead of a player, and never asks for its files", async () => {
    const runsApi = { artifacts: vi.fn(), series: vi.fn(), summary: vi.fn(), compare: vi.fn() } as unknown as RunsApi;
    const videoApi = { listRuns: async () => [failed], runFileUrl: () => "" } as unknown as VideoApi;
    render(<RunsView videoApi={videoApi} api={runsApi} runId={failed.id} compareId={null} onRoute={() => undefined} onReopen={async () => undefined} onRerun={async () => undefined} />);
    expect(await screen.findByText(/The job failed before writing any file/)).toBeInTheDocument();
    expect(screen.getByLabelText("Log tail")).toHaveTextContent("loading weights");
    expect(document.querySelector("video")).toBeNull();
    expect(screen.getByRole("button", { name: /Rerun/ })).toBeEnabled();
    expect(runsApi.artifacts).not.toHaveBeenCalled();
    expect(runsApi.series).not.toHaveBeenCalled();
  });

  it("deep-links a comparison by ids, dots included", () => {
    const route = { view: "runs" as const, runId: "20260101_000000_a.v1", compareId: "20260101_000001_b" };
    expect(routeHash(route)).toBe("#/runs/20260101_000000_a.v1/vs/20260101_000001_b");
    expect(parseHash(routeHash(route))).toEqual(route);
    expect(parseHash("#/runs/x%2Fy")).toEqual({ view: "runs", runId: "x/y", compareId: null });
  });
});

describe("RunsView delete", () => {
  const mk = (id: string, video: string): RunRecord => ({ ...failed, id, video, status: "done", complete: true, error: null, log_tail: undefined });
  const A = mk("20260101_000000_a", "a.mp4");
  const B = mk("20260101_000001_b", "b.mp4");

  function setup(opts: { runId?: string | null; compareId?: string | null; remove?: () => Promise<void>; list?: RunRecord[] } = {}) {
    let list = opts.list ?? [A, B];
    const remove = vi.fn(opts.remove ?? (async () => undefined));
    const size = vi.fn(async () => 1536);
    const runsApi = { artifacts: async () => [], series: async () => ({ track: null, found: false, tracks: [], t: [], raw: [], smoothed: null, video_t: [], points: 0, downsampled: false }), summary: async () => ({ summary: {}, metrics: [] }), compare: async () => { throw new Error("no comparison"); }, size, remove: async (id: string) => { await remove(); list = list.filter((r) => r.id !== id); } } as unknown as RunsApi;
    const videoApi = { listRuns: async () => list, runFileUrl: () => "" } as unknown as VideoApi;
    const onRoute = vi.fn();
    render(<RunsView videoApi={videoApi} api={runsApi} runId={opts.runId ?? null} compareId={opts.compareId ?? null} onRoute={onRoute} onReopen={async () => undefined} onRerun={async () => undefined} />);
    return { remove, size, onRoute };
  }

  it("deletes a run after confirming", async () => {
    const user = userEvent.setup();
    const { remove, size } = setup();
    await user.click(await screen.findByRole("button", { name: "Delete a.mp4" }));
    expect(await screen.findByText(/Delete a\.mp4 permanently\? This frees 1\.5 KB\./)).toBeInTheDocument();
    expect(size).toHaveBeenCalledWith(A.id);
    await user.click(within(screen.getByRole("dialog")).getByRole("button", { name: "Delete" }));
    await waitFor(() => expect(remove).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(screen.queryByRole("button", { name: "Delete a.mp4" })).toBeNull());
    expect(screen.getByRole("button", { name: "Delete b.mp4" })).toBeInTheDocument();
  });

  it("cancel leaves the run", async () => {
    const user = userEvent.setup();
    const { remove } = setup();
    await user.click(await screen.findByRole("button", { name: "Delete a.mp4" }));
    await user.click(await screen.findByRole("button", { name: "Cancel" }));
    expect(remove).not.toHaveBeenCalled();
    expect(screen.getByRole("button", { name: "Delete a.mp4" })).toBeInTheDocument();
  });

  it("shows the server error and stays open on 409", async () => {
    const user = userEvent.setup();
    setup({ remove: async () => { throw new ApiError("run is still queued or running", 409); } });
    await user.click(await screen.findByRole("button", { name: "Delete a.mp4" }));
    await user.click(await screen.findByRole("button", { name: "Delete" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("run is still queued or running");
    expect(screen.getByRole("dialog")).toBeInTheDocument();
  });

  it("deleting the selected run clears the route", async () => {
    const user = userEvent.setup();
    const { onRoute } = setup({ runId: A.id, compareId: B.id });
    await user.click((await screen.findAllByRole("button", { name: "Delete a.mp4" }))[0]);
    await user.click(await screen.findByRole("button", { name: "Delete" }));
    await waitFor(() => expect(onRoute).toHaveBeenCalledWith(null, null));
  });

  it("deleting the compare run clears only the comparison", async () => {
    const user = userEvent.setup();
    const { onRoute } = setup({ runId: A.id, compareId: B.id });
    await user.click((await screen.findAllByRole("button", { name: "Delete b.mp4" }))[0]);
    await user.click(await screen.findByRole("button", { name: "Delete" }));
    await waitFor(() => expect(onRoute).toHaveBeenCalledWith(A.id, null));
  });

  it("treats a 404 on delete as already deleted", async () => {
    const user = userEvent.setup();
    setup({ remove: async () => { throw new ApiError("unknown run", 404); } });
    await user.click(await screen.findByRole("button", { name: "Delete a.mp4" }));
    await user.click(await screen.findByRole("button", { name: "Delete" }));
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("disables Cancel and ignores Escape while deleting", async () => {
    const user = userEvent.setup();
    let release: () => void = () => undefined;
    const { remove } = setup({ remove: () => new Promise<void>((r) => { release = r; }) });
    await user.click(await screen.findByRole("button", { name: "Delete a.mp4" }));
    await user.click(await screen.findByRole("button", { name: "Delete" }));
    await waitFor(() => expect(remove).toHaveBeenCalledTimes(1));
    expect(screen.getByRole("button", { name: "Cancel" })).toBeDisabled();
    fireEvent(screen.getByRole("dialog"), new Event("cancel", { cancelable: true }));
    expect(screen.getByRole("dialog")).toBeInTheDocument();
    release();
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
  });

  it("deletes from the detail panel's trash button", async () => {
    const user = userEvent.setup();
    const { remove } = setup({ runId: A.id });
    const buttons = await screen.findAllByRole("button", { name: "Delete a.mp4" });
    const detail = buttons.find((b) => b.getAttribute("title") === "Delete this run");
    expect(detail).toBeDefined();
    await user.click(detail!);
    await user.click(within(await screen.findByRole("dialog")).getByRole("button", { name: "Delete" }));
    await waitFor(() => expect(remove).toHaveBeenCalledTimes(1));
  });

  it("skips the size call for a failed job entry", async () => {
    const user = userEvent.setup();
    const { size, remove } = setup({ list: [failed] });
    await user.click(await screen.findByRole("button", { name: "Delete road.mkv" }));
    expect(await screen.findByText(/Delete road\.mkv permanently\?/)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Delete" }));
    await waitFor(() => expect(remove).toHaveBeenCalledTimes(1));
    expect(size).not.toHaveBeenCalled();
  });
});

describe("formatBytes", () => {
  it("formats sizes", () => {
    expect(formatBytes(0)).toBe("0 B");
    expect(formatBytes(1536)).toBe("1.5 KB");
    expect(formatBytes(1.2 * 1024 ** 3)).toBe("1.2 GB");
  });
});
