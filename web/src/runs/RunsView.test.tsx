import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { RunsApi } from "../api/runs";
import type { RunRecord, VideoApi } from "../api/video";
import { parseHash, routeHash } from "../ui/route";
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
