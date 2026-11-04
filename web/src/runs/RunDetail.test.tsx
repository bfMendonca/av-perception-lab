import { render, screen, waitFor, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { MetricInfo, RunsApi, Series, TrackInfo } from "../api/runs";
import type { RunRecord } from "../api/video";
import { pickDefaultTrack, RunDetail, SHARED_ID_LABEL, TRACK_ID_NOTE } from "./RunDetail";

const track = (id: number, rows: number, t0: number, t1: number): TrackInfo => ({ id, rows, t0, t1, class_name: "vehicle" });

// Test data only: the component knows no run, track or metric names.
const tracksB = [track(7, 500, 40, 90), track(3, 200, 0, 20), track(9, 50, 5, 8)];
const tracksA = [track(7, 300, 0, 30), track(3, 150, 2, 25), track(5, 80, 0, 10)];

describe("pickDefaultTrack", () => {
  it("prefers a shared id whose time ranges overlap over a longer non-overlapping one", () => {
    // 7 is longer and in A too, but A's #7 lives in 0-30 s, B's in 40-90 s
    expect(pickDefaultTrack(tracksB, tracksA)).toBe(3);
  });

  it("falls back to the longest shared id, then to none", () => {
    expect(pickDefaultTrack(tracksB, [track(7, 10, 0, 1), track(9, 10, 100, 110)])).toBe(7);
    expect(pickDefaultTrack(tracksB, [track(42, 10, 0, 100)])).toBeNull();
  });
});

const run = (id: string): RunRecord => ({
  id,
  created: "2026-10-31T10:00:00",
  video: `${id}.mkv`,
  detectors: ["det"],
  depth: "depth",
  frames: 100,
  wall_s: 10,
  status: "done",
  complete: true,
  has_config: true,
  error: null,
});

const metrics: MetricInfo[] = [{ key: "jitter_m", label: "Jitter", unit: "m", better: "lower" }];

function seriesOf(tracks: TrackInfo[], id: number | null | undefined): Series {
  const t = tracks.find((x) => x.id === id) ?? tracks[0];
  return { track: t.id, found: true, tracks, t: [t.t0, t.t1], raw: [10, 11], smoothed: [10, 10.5], video_t: [0, 1], points: 2, downsampled: false };
}

function fakeApi(): RunsApi & { seriesCalls: [string, number | null | undefined][] } {
  const seriesCalls: [string, number | null | undefined][] = [];
  return {
    seriesCalls,
    summary: async () => ({ summary: { jitter_m: 0.4 }, metrics }),
    compare: async (a, b) => ({ a, b, metrics, summary_a: { jitter_m: 0.5 }, summary_b: { jitter_m: 0.4 }, deltas: { jitter_m: { abs: -0.1, rel: -0.2 } }, config_diff: [], config_origin: { a: "config.yaml", b: "config.yaml" } }),
    series: async (id, trackId) => {
      seriesCalls.push([id, trackId]);
      return seriesOf(id === "B" ? tracksB : tracksA, trackId);
    },
    artifacts: async () => [],
    fetchJson: async () => ({}),
    reopen: async () => ({ config: {}, extras: {}, origin: "config.yaml", file_name: "x.yaml" }),
    rerun: async () => "job",
    size: async () => 0,
    remove: async () => undefined,
  };
}

const noop = async () => undefined;

describe("RunDetail track ids", () => {
  it("in compare mode labels shared ids without claiming the same object, notes it, and picks the overlapping track", async () => {
    const api = fakeApi();
    render(<RunDetail run={run("B")} against={run("A")} runs={[run("A"), run("B")]} api={api} onCompare={() => undefined} onReopen={noop} onRerun={noop} onDelete={() => undefined} />);
    const select = (await screen.findByRole("combobox", { name: "Track" })) as HTMLSelectElement;
    await waitFor(() => expect(select.value).toBe("3"));
    const option3 = within(select).getByRole("option", { name: new RegExp(`^#3 .*${SHARED_ID_LABEL}$`) });
    expect(option3).toBeInTheDocument();
    expect(within(select).getByRole("option", { name: /^#9 / }).textContent).not.toContain(SHARED_ID_LABEL);
    expect(select.textContent).not.toMatch(/in both runs/);
    expect(screen.getByTestId("track-id-note")).toHaveTextContent(TRACK_ID_NOTE);
    await waitFor(() => expect(api.seriesCalls).toContainEqual(["A", 3]));
  });

  it("shows no track-id note and no shared label without a comparison run", async () => {
    render(<RunDetail run={run("B")} against={null} runs={[run("B")]} api={fakeApi()} onCompare={() => undefined} onReopen={noop} onRerun={noop} onDelete={() => undefined} />);
    const select = (await screen.findByRole("combobox", { name: "Track" })) as HTMLSelectElement;
    expect(select.value).toBe("7"); // B's own default: its longest track
    expect(select.textContent).not.toContain(SHARED_ID_LABEL);
    expect(screen.queryByTestId("track-id-note")).toBeNull();
    expect(screen.queryByText(TRACK_ID_NOTE)).toBeNull();
  });
});
