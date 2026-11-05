import { describe, expect, it } from "vitest";
import type { ConfigDict } from "../api/types";
import { mcapRecord, mcapTopics, videoRecord } from "../test/fixtures";
import { bindMcap, cameraInfoAvailability, needsBinding, rectifyAvailability, setRectify, setUseCameraInfo, unbindMcap } from "./mcapCamera";

const [depth, cam, front, rect] = mcapTopics.images;

describe("mcapCamera", () => {
  it("rectifyAvailability follows the topic's CameraInfo", () => {
    expect(rectifyAvailability(cam)).toEqual({ state: "available" });
    expect(rectifyAvailability(front)).toEqual({ state: "no_camera_info", reason: "no CameraInfo for this topic" });
    expect(rectifyAvailability(rect)).toEqual({ state: "already_rectified", reason: "already rectified" });
    expect(rectifyAvailability({ ...cam, hasDistortion: false, cameraInfoError: "size 8x6 != 16x12" })).toEqual({ state: "unsupported", reason: "size 8x6 != 16x12" });
    expect(rectifyAvailability(depth).state).toBe("unsupported");
    expect(rectifyAvailability(depth).reason).toContain("16UC1");
    expect(rectifyAvailability(null)).toEqual({ state: "unsupported", reason: "pick an image topic" });
  });

  it("cameraInfoAvailability needs a readable CameraInfo (distortion or not)", () => {
    expect(cameraInfoAvailability(cam)).toEqual({ available: true });
    expect(cameraInfoAvailability(rect)).toEqual({ available: true });
    expect(cameraInfoAvailability(front)).toEqual({ available: false, reason: "no CameraInfo for this topic" });
    expect(cameraInfoAvailability({ ...cam, cameraInfoError: "bad" })).toEqual({ available: false, reason: "bad" });
  });

  it("bindMcap picks a supported topic, drops source.video and resets what the topic cannot do", () => {
    const base: ConfigDict = { source: { video: "a.mp4", stride: 2 }, camera: { calibration: { from_source: true } }, rectify: { enabled: true, alpha: 0.5 } };
    expect(needsBinding(base, mcapRecord)).toBe(true);
    const bound = bindMcap(base, mcapRecord);
    expect(bound.source).toEqual({ mcap: mcapRecord.original, topic: "/cam/image_raw", stride: 2 });
    expect(bound.rectify).toEqual({ enabled: true, alpha: 0.5 });
    expect(needsBinding(bound, mcapRecord)).toBe(false);

    const front_ = bindMcap(bound, mcapRecord, front.name);
    expect(front_.source).toEqual({ mcap: mcapRecord.original, topic: front.name, stride: 2 });
    expect(front_.rectify).toBeUndefined();
    expect(front_.camera).toEqual({});
    // a topic of the record is kept on a rebind; an unsupported one is not
    expect(bindMcap({ source: { mcap: "/elsewhere.mcap", topic: rect.name } }, mcapRecord).source).toEqual({ mcap: mcapRecord.original, topic: rect.name });
    expect(bindMcap({ source: { topic: depth.name } }, mcapRecord).source).toEqual({ mcap: mcapRecord.original, topic: cam.name });
  });

  it("unbindMcap clears the MCAP-only keys", () => {
    const cfg: ConfigDict = { source: { mcap: "/x.mcap", topic: "/t", stride: 1 }, camera: { focal_px: null, calibration: { from_source: true, cross_check: true } }, rectify: { enabled: true } };
    expect(unbindMcap(cfg)).toEqual({ source: { stride: 1 }, camera: { focal_px: null } });
    expect(needsBinding({ source: { video: "a.mp4" } }, videoRecord)).toBe(false);
  });

  it("setUseCameraInfo and setRectify remove the exclusive intrinsics sources", () => {
    const cfg: ConfigDict = { camera: { focal_px: 900, estimate: { frames: 8 }, calibration: { path: "c.yaml", undistort: true, cross_check: true } } };
    expect(setUseCameraInfo(cfg, true).camera).toEqual({ estimate: { frames: 8 }, calibration: { from_source: true, cross_check: true } });
    expect(setUseCameraInfo(setUseCameraInfo(cfg, true), false).camera).toEqual({ estimate: { frames: 8 } });
    const on = setRectify(cfg, true, { enabled: false, alpha: 0 });
    expect(on.rectify).toEqual({ enabled: true, alpha: 0 });
    expect(on.camera).toEqual({ estimate: { frames: 8 } });
    // a from_source calibration stays with rectify
    expect(setRectify({ camera: { calibration: { from_source: true } } }, true).camera).toEqual({ calibration: { from_source: true } });
    expect(setRectify(on, false).rectify).toBeUndefined();
  });
});
