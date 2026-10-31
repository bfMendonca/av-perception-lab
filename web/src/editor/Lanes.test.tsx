import { render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { App } from "../App";
import { fakeClient, makeCatalog, sampleConfig } from "../test/fixtures";
import { registerStageView } from "./registry";
import { createEditorStore } from "./yamlState";

const unregister: (() => void)[] = [];
afterEach(() => unregister.splice(0).forEach((u) => u()));

async function renderWith(catalog = makeCatalog(), config = sampleConfig.config) {
  const client = fakeClient(catalog, { config, extras: {} });
  const store = createEditorStore(client, { debounceMs: 0 });
  await store.load("ignored", "x.yaml");
  render(<App client={client} store={store} />);
}

describe("lanes", () => {
  it("registerStageView replaces the generic lane for its yaml_key", async () => {
    unregister.push(registerStageView("aggregation", ({ stage, value }) => <div data-testid="custom-lane">{`${stage.title}: ${JSON.stringify(value)}`}</div>));
    await renderWith();
    expect(await screen.findByTestId("custom-lane")).toHaveTextContent('Aggregation: {"type":"merge"}');
    expect(screen.queryByRole("region", { name: "Aggregation" })).not.toBeInTheDocument();
    expect(screen.getByRole("region", { name: "Detectors" })).toBeInTheDocument(); // others stay generic
  });

  it("consecutive stages sharing a group render as one lane with both stages' cards", async () => {
    const base = makeCatalog();
    const catalog = makeCatalog({
      stages: [
        ...base.stages.filter((s) => s.yaml_key !== "smoothing"),
        { yaml_key: "follow", title: "Follower", kind: "detector", multiple: false, optional: true, order: 60, general: false, has_section: false, group: "Track group" },
        { ...base.stages.find((s) => s.yaml_key === "smoothing")!, order: 70, group: "Track group" },
        { yaml_key: "planners", title: "Path planners", kind: "planner", multiple: true, optional: true, order: 90, general: false, has_section: false, group: null },
      ],
    });
    await renderWith(catalog, { ...sampleConfig.config, follow: { type: "beta" }, planners: [{ type: "lattice" }] });
    const group = await screen.findByRole("region", { name: "Track group" });
    expect(within(group).getByRole("button", { name: /beta/i })).toBeInTheDocument();
    expect(within(group).getByRole("button", { name: /smoothing/i })).toBeInTheDocument();
    expect(within(group).getByRole("region", { name: "Follower" })).toBeInTheDocument();
    expect(screen.queryByRole("region", { name: "Smoothing" })).toBe(within(group).getByRole("region", { name: "Smoothing" }));
    // an ungrouped unknown stage still renders alone
    const planners = screen.getByRole("region", { name: "Path planners" });
    expect(group).not.toContainElement(planners);
    expect(within(planners).getByRole("button", { name: /lattice/i })).toBeInTheDocument();
  });
});
