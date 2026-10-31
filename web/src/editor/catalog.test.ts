import { describe, expect, it } from "vitest";
import { makeCatalog, thresholdField } from "../test/fixtures";
import { generalStages, laneStages, refineErrors, summarize } from "./catalog";
import { diffTrees, isUnder, setIn, toDotted } from "./paths";

describe("catalog and path helpers", () => {
  it("dotted paths match the server's error format", () => {
    expect(toDotted(["detectors", 0, "threshold"])).toBe("detectors[0].threshold");
    expect(isUnder("detectors[0].threshold", "detectors[0]")).toBe(true);
    expect(isUnder("detectors[10]", "detectors[1]")).toBe(false);
  });

  it("setIn is immutable and undefined removes the key", () => {
    const a = { s: { x: 1, y: 2 } };
    const b = setIn(a, ["s", "x"], undefined);
    expect(b).toEqual({ s: { y: 2 } });
    expect(a).toEqual({ s: { x: 1, y: 2 } });
  });

  it("diffTrees reports added and removed list entries", () => {
    expect(diffTrees({ d: [{ t: "a" }] }, { d: [{ t: "a" }, { t: "b" }] })).toEqual([{ path: "d[1]", before: undefined, after: { t: "b" } }]);
  });

  it("lanes and general settings follow the catalog order and flags", () => {
    const c = makeCatalog();
    expect(laneStages(c).map((s) => s.yaml_key)).toEqual(["source", "detectors", "aggregation", "smoothing"]);
    expect(generalStages(c).map((s) => s.yaml_key)).toEqual(["render"]);
  });

  it("refineErrors attaches a spec-level domain error to the field it names", () => {
    const errors = [
      { path: "detectors[0]", message: "threshold must be in (0, 1], got 7.0" },
      { path: "detectors[0]", message: "something else" },
    ];
    expect(refineErrors(errors, "detectors[0]", [thresholdField]).map((e) => e.path)).toEqual(["detectors[0].threshold", "detectors[0]"]);
  });

  it("summarize prefers labelled fields and skips non-scalars", () => {
    expect(summarize([thresholdField], { type: "x", threshold: 0.5, classes: ["a"], extra: 3 })).toBe("threshold 0.5 · extra 3");
  });
});
