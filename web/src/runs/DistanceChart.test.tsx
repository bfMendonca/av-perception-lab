import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { Series } from "../api/runs";
import { DistanceChart, PAD, makeScale, nearestIndex } from "./DistanceChart";

const series = (over: Partial<Series> = {}): Series => ({
  track: 3,
  found: true,
  tracks: [{ id: 3, rows: 3, t0: 10, t1: 20, class_name: "vehicle" }],
  t: [10, 15, 20],
  raw: [30, 28, 26],
  smoothed: [29.5, 28, 26.5],
  // the clip started at 10 s: the annotated video starts at 0
  video_t: [0, 5, 10],
  points: 3,
  downsampled: false,
  ...over,
});

describe("DistanceChart", () => {
  it("DistanceChart maps seconds to x and seeks on click", () => {
    const x = makeScale([10, 20], [PAD.left, PAD.left + 100]);
    expect(x(10)).toBe(PAD.left);
    expect(x(15)).toBe(PAD.left + 50);
    expect(x.invert(PAD.left + 100)).toBe(20);
    expect(nearestIndex([10, 15, 20], 16.4)).toBe(1);

    const onSeek = vi.fn();
    const { container } = render(<DistanceChart b={series()} a={series({ smoothed: null, raw: [31, 29, 27] })} labelB="B" labelA="A" onSeek={onSeek} />);
    const svg = container.querySelector("svg") as SVGSVGElement;
    // jsdom has no layout: the chart is 800 px wide at its default size
    svg.getBoundingClientRect = () => ({ left: 0, top: 0, width: 800, height: 260, right: 800, bottom: 260, x: 0, y: 0, toJSON: () => ({}) });
    const plotW = 800 - PAD.left - PAD.right;
    const plot = screen.getByTestId("chart-plot");
    // a click 52 % of the way along the time axis lands nearest t = 15 s,
    // which is 5 s into the annotated video
    fireEvent.click(plot, { clientX: PAD.left + plotW * 0.52 });
    expect(onSeek).toHaveBeenCalledWith(5);
    fireEvent.click(plot, { clientX: PAD.left + plotW });
    expect(onSeek).toHaveBeenLastCalledWith(10);
    // three lines: A (dashed), B raw (grey) and B smoothed
    expect(container.querySelectorAll("path.chart__line")).toHaveLength(3);
    expect(container.querySelector("path.chart__line--raw")).not.toBeNull();
    // the path's first point sits at x(10 s) = the left edge of the plot
    expect(container.querySelector("path.chart__line--b")?.getAttribute("d")).toMatch(new RegExp(`^M${PAD.left}\\.0 `));
  });

  it("keyboard cursor seeks with Enter and a run without smoothing draws raw only", () => {
    const onSeek = vi.fn();
    const { container } = render(<DistanceChart b={series({ smoothed: null })} labelB="B" onSeek={onSeek} />);
    const plot = screen.getByTestId("chart-plot");
    fireEvent.keyDown(plot, { key: "ArrowRight" });
    fireEvent.keyDown(plot, { key: "ArrowRight" });
    expect(screen.getByRole("status")).toHaveTextContent("15.0 s");
    fireEvent.keyDown(plot, { key: "Enter" });
    expect(onSeek).toHaveBeenCalledWith(5);
    expect(container.querySelectorAll("path.chart__line")).toHaveLength(1);
  });

  it("says when there is nothing to draw", () => {
    render(<DistanceChart b={series({ found: false, t: [], raw: [], smoothed: null, video_t: [] })} labelB="B" onSeek={() => undefined} />);
    expect(screen.getByText("This track is not in the run.")).toBeInTheDocument();
  });
});
