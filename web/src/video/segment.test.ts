import { describe, expect, it } from "vitest";
import { buildRunConfig, estimateSeconds, expectedFrames, formatDuration, formatTime, segmentToSource } from "./segment";

describe("segment math", () => {
  it("segmentToSource clamps duration to the video end", () => {
    expect(segmentToSource(8, 5, 1, 10)).toEqual({ start: 8, duration: 2, stride: 1 });
    expect(segmentToSource(-3, 4, 2, 10)).toEqual({ start: 0, duration: 4, stride: 2 });
    // start past the end is pulled back inside the video
    expect(segmentToSource(30, 3, 1, 10).start).toBeLessThan(10);
    expect(segmentToSource(30, 3, 1, 10).duration).toBeGreaterThan(0);
    // 0 = to the end; unknown video length leaves the duration alone
    expect(segmentToSource(2, 0, 1, 10)).toEqual({ start: 2, duration: 0, stride: 1 });
    expect(segmentToSource(2, 7.12345, 0, undefined)).toEqual({ start: 2, duration: 7.123, stride: 1 });
    expect(segmentToSource(1, 2, 2.6, 10).stride).toBe(3);
  });

  it("estimateSeconds uses stride", () => {
    // 3 s at 25 fps: 75 frames at stride 1, 25 at stride 3
    expect(expectedFrames(3, 25, 1)).toBe(75);
    expect(expectedFrames(3, 25, 3)).toBe(25);
    expect(estimateSeconds(expectedFrames(3, 25, 1))).toBe(37.5);
    expect(estimateSeconds(expectedFrames(3, 25, 3))).toBe(12.5);
    expect(estimateSeconds(expectedFrames(3, 25, 3), 5)).toBe(5);
    expect(expectedFrames(0, 25, 1)).toBe(0);
  });

  it("formats times and estimates", () => {
    expect(formatTime(65.34)).toBe("1:05.3");
    expect(formatTime(3723)).toBe("1:02:03.0");
    expect(formatDuration(37.5)).toBe("~38 s");
    expect(formatDuration(125)).toBe("~2 min 5 s");
    expect(formatDuration(120)).toBe("~2 min");
  });

  it("buildRunConfig fills the source, keeping everything else (output too)", () => {
    const config = { source: { video: "old.mp4", max_frames: 5 }, output: { out_root: "x" }, detectors: [{ type: "a" }] };
    const built = buildRunConfig(config, "/lib/v.mkv", { start: 1, duration: 3, stride: 2 });
    expect(built).toEqual({
      source: { video: "/lib/v.mkv", max_frames: 5, start: 1, duration: 3, stride: 2 },
      output: { out_root: "x" },
      detectors: [{ type: "a" }],
    });
    expect(config.source.video).toBe("old.mp4");
  });
});
