import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { App } from "../App";
import { fakeClient, makeCatalog, sampleConfig } from "../test/fixtures";
import { createEditorStore } from "./yamlState";

describe("editor view", () => {
  it("unknown stage from the catalog renders as a generic lane with a card per module", async () => {
    const base = makeCatalog();
    const catalog = makeCatalog({
      stages: [
        ...base.stages,
        { yaml_key: "planners", title: "Path planners", kind: "planner", multiple: true, optional: true, order: 90, general: false, has_section: false },
      ],
      modules: { ...base.modules, planner: { lattice: { doc: "Lattice planner.", schema: { fields: [] } } } },
    });
    const parsed = {
      config: { ...sampleConfig.config, planners: [{ type: "lattice", horizon: 4 }, { type: "lattice", horizon: 8 }] },
      extras: {},
    };
    const client = fakeClient(catalog, parsed);
    const store = createEditorStore(client, { debounceMs: 0 });
    await store.load("ignored", "p.yaml");
    render(<App client={client} store={store} />);
    const lane = await screen.findByRole("region", { name: "Path planners" });
    expect(within(lane).getAllByRole("button", { name: /lattice/i })).toHaveLength(2);
    expect(within(lane).getByRole("button", { name: /add/i })).toBeInTheDocument();
  });

  it("catalog_version greater than the supported one shows a non-blocking banner and still renders", async () => {
    const client = fakeClient(makeCatalog({ catalog_version: 99 }));
    const store = createEditorStore(client, { debounceMs: 0 });
    await store.load("ignored", "s.yaml");
    render(<App client={client} store={store} />);
    expect(await screen.findByRole("status", { name: /catalog version/i })).toHaveTextContent("99");
    expect(screen.getByRole("region", { name: "Detectors" })).toBeInTheDocument();
  });

  it("server validation errors attach to the field by dotted path", async () => {
    const client = fakeClient();
    client.errors = [{ path: "detectors[0].threshold", message: "must be between 0 and 1" }];
    const store = createEditorStore(client, { debounceMs: 0 });
    await store.load("ignored", "s.yaml");
    render(<App client={client} store={store} />);
    const lane = await screen.findByRole("region", { name: "Detectors" });
    await userEvent.click(within(lane).getByRole("button", { name: /alpha/i }));
    const field = await screen.findByTestId("field:detectors[0].threshold");
    await waitFor(() => expect(within(field).getByText("must be between 0 and 1")).toBeInTheDocument());
    expect(within(field).getByRole("slider")).toHaveAttribute("aria-invalid", "true");
  });
});
