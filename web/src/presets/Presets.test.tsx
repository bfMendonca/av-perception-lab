import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { App } from "../App";
import { ApiError } from "../api/client";
import type { PresetInfo, PresetsApi } from "../api/presets";
import type { ParsedConfig } from "../api/types";
import { createEditorStore } from "../editor/yamlState";
import { fakeClient, makeCatalog, sampleConfig } from "../test/fixtures";

const BUILTIN: PresetInfo = { id: "fast", name: "Fast", description: "Light detector, quick.", kind: "builtin" };
const MINE: PresetInfo = { id: "my-mix", name: "My mix", description: "Two detectors merged.", kind: "user" };

const presetConfig: ParsedConfig = {
  config: {
    source: { video: "preset.mp4", stride: 4 },
    detectors: [{ type: "beta", threshold: 0.9 }],
    aggregation: { type: "merge" },
    render: { hud: false },
  },
  extras: {},
};

function fakePresets(initial: PresetInfo[] = [BUILTIN, MINE]) {
  const api = {
    items: [...initial],
    saved: [] as Array<{ name: string; overwrite: boolean }>,
    list: vi.fn(async () => [...api.items]),
    get: vi.fn(async (_id: string) => presetConfig),
    save: vi.fn(async (name: string, _c: unknown, _e: unknown, overwrite: boolean) => {
      api.saved.push({ name, overwrite });
      return { id: name.toLowerCase(), name, description: "", kind: "user" as const };
    }),
    remove: vi.fn(async (id: string) => {
      api.items = api.items.filter((p) => p.id !== id);
    }),
  };
  return api satisfies PresetsApi & Record<string, unknown>;
}

async function setup(presets = fakePresets()) {
  const client = fakeClient(makeCatalog(), sampleConfig);
  const store = createEditorStore(client, { debounceMs: 0 });
  await store.load("ignored", "pipeline.yaml");
  render(<App client={client} store={store} presetsApi={presets} />);
  await screen.findByRole("region", { name: "Detectors" });
  return { presets, store };
}

const openMenu = async () => {
  await userEvent.click(screen.getByRole("button", { name: /^presets$/i }));
  return screen.findByRole("dialog", { name: "Presets" });
};

afterEach(() => vi.restoreAllMocks());

describe("presets menu", () => {
  it("lists built-in and user presets in two groups with name and description", async () => {
    await setup();
    const menu = await openMenu();
    const builtin = await within(menu).findByRole("group", { name: "Built-in" });
    const mine = within(menu).getByRole("group", { name: "Mine" });
    expect(within(builtin).getByText("Fast")).toBeInTheDocument();
    expect(within(builtin).getByText("Light detector, quick.")).toBeInTheDocument();
    expect(within(mine).getByText("My mix")).toBeInTheDocument();
    expect(within(mine).getByText("Two detectors merged.")).toBeInTheDocument();
  });

  it("picking a preset loads it and keeps the video and output of the editor", async () => {
    const { presets, store } = await setup();
    const menu = await openMenu();
    await userEvent.click(await within(menu).findByRole("button", { name: /Fast/ }));
    await waitFor(() => expect(store.getState().pipeline.config.detectors).toEqual(presetConfig.config.detectors));
    expect(presets.get).toHaveBeenCalledWith("fast");
    expect(store.getState().pipeline.config.source).toEqual(sampleConfig.config.source);
    expect(store.getState().dirty).toBe(false);
    expect(screen.queryByRole("dialog", { name: "Presets" })).not.toBeInTheDocument();
  });

  it("picking a preset with unsaved changes asks first and does nothing when declined", async () => {
    const { presets, store } = await setup();
    const before = store.getState().pipeline.config;
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(false);
    store.setOption(["render", "hud"], false);
    expect(store.getState().dirty).toBe(true);
    const kept = store.getState().pipeline.config;
    const menu = await openMenu();
    await userEvent.click(await within(menu).findByRole("button", { name: /Fast/ }));
    expect(confirm).toHaveBeenCalledWith("Discard the unsaved changes and open preset Fast?");
    expect(presets.get).not.toHaveBeenCalled();
    expect(store.getState().pipeline.config.detectors).toEqual(before.detectors);
    expect(store.getState().pipeline.config).toBe(kept);
  });

  it("save as preset asks for a name; a 409 offers overwrite or another name, and overwrite sends overwrite=true", async () => {
    const presets = fakePresets();
    presets.save.mockRejectedValueOnce(new ApiError("A preset named 'my-mix' already exists", 409));
    await setup(presets);
    await userEvent.click(screen.getByRole("button", { name: /save as preset/i }));
    const dialog = await screen.findByRole("dialog", { name: "Save as preset" });
    await userEvent.type(within(dialog).getByLabelText("Name"), "My mix");
    await userEvent.click(within(dialog).getByRole("button", { name: "Save" }));
    expect(await within(dialog).findByText(/already exists/)).toBeInTheDocument();
    expect(presets.saved).toEqual([]);
    await userEvent.click(within(dialog).getByRole("button", { name: "Overwrite" }));
    await waitFor(() => expect(presets.saved).toEqual([{ name: "My mix", overwrite: true }]));
    await waitFor(() => expect(screen.queryByRole("dialog", { name: "Save as preset" })).not.toBeInTheDocument());
    expect(await screen.findByText("Saved preset My mix.")).toBeInTheDocument();
    expect(presets.list.mock.calls.length).toBeGreaterThanOrEqual(2);
  });

  it("an empty or symbols-only name shows the server message and keeps the dialog open", async () => {
    const presets = fakePresets();
    presets.save.mockRejectedValueOnce(new ApiError("The name has no letters or digits", 422));
    await setup(presets);
    await userEvent.click(screen.getByRole("button", { name: /save as preset/i }));
    const dialog = await screen.findByRole("dialog", { name: "Save as preset" });
    await userEvent.type(within(dialog).getByLabelText("Name"), "!!!");
    await userEvent.click(within(dialog).getByRole("button", { name: "Save" }));
    expect(await within(dialog).findByText("The name has no letters or digits")).toBeInTheDocument();
    expect(screen.getByRole("dialog", { name: "Save as preset" })).toBeInTheDocument();
    expect(within(dialog).queryByRole("button", { name: "Overwrite" })).not.toBeInTheDocument();
  });

  it("a built-in name clash (422) shows the message without an Overwrite button and keeps the dialog open", async () => {
    const presets = fakePresets();
    presets.save.mockRejectedValueOnce(new ApiError("'Full stack' is the name of a built-in preset; choose another name.", 422));
    await setup(presets);
    await userEvent.click(screen.getByRole("button", { name: /save as preset/i }));
    const dialog = await screen.findByRole("dialog", { name: "Save as preset" });
    await userEvent.type(within(dialog).getByLabelText("Name"), "Full stack");
    await userEvent.click(within(dialog).getByRole("button", { name: "Save" }));
    expect(await within(dialog).findByText(/name of a built-in preset/)).toBeInTheDocument();
    expect(within(dialog).queryByRole("button", { name: "Overwrite" })).not.toBeInTheDocument();
    expect(within(dialog).getByRole("button", { name: "Save" })).toBeInTheDocument();
    expect(screen.getByRole("dialog", { name: "Save as preset" })).toBeInTheDocument();
  });

  it("opening the menu refreshes the list", async () => {
    const presets = fakePresets();
    await setup(presets);
    await waitFor(() => expect(presets.list).toHaveBeenCalledTimes(1));
    presets.items = [BUILTIN, { ...MINE, id: "late", name: "Late arrival" }];
    const menu = await openMenu();
    expect(await within(menu).findByText("Late arrival")).toBeInTheDocument();
    expect(presets.list).toHaveBeenCalledTimes(2);
  });

  it("a failed pick (stale 404) refreshes the list", async () => {
    const presets = fakePresets();
    await setup(presets);
    const menu = await openMenu();
    await within(menu).findByText("Fast");
    const before = presets.list.mock.calls.length;
    presets.get.mockRejectedValueOnce(new ApiError("No preset", 404));
    await userEvent.click(within(menu).getByRole("button", { name: /^My mix/ }));
    expect(await screen.findByText(/Could not open preset My mix/)).toBeInTheDocument();
    await waitFor(() => expect(presets.list.mock.calls.length).toBeGreaterThan(before));
  });

  it("delete asks for confirmation, removes the row, and a 404 (already gone) just refreshes the list", async () => {
    const ghost: PresetInfo = { ...MINE, id: "ghost", name: "Ghost" };
    const presets = fakePresets([BUILTIN, MINE, ghost]);
    await setup(presets);
    const menu = await openMenu();
    const confirm = vi.spyOn(window, "confirm").mockReturnValueOnce(false).mockReturnValue(true);
    await userEvent.click(await within(menu).findByRole("button", { name: "Delete preset My mix" }));
    expect(confirm).toHaveBeenCalledWith("Delete the preset My mix?");
    expect(presets.remove).not.toHaveBeenCalled();
    await userEvent.click(within(menu).getByRole("button", { name: "Delete preset My mix" }));
    await waitFor(() => expect(within(menu).queryByText("My mix")).not.toBeInTheDocument());
    expect(presets.remove).toHaveBeenCalledWith("my-mix");

    // already gone on the server: 404 only refreshes
    presets.remove.mockImplementationOnce(async () => {
      presets.items = [BUILTIN];
      throw new ApiError("Not found", 404);
    });
    await userEvent.click(await within(menu).findByRole("button", { name: "Delete preset Ghost" }));
    await waitFor(() => expect(within(menu).queryByText("Ghost")).not.toBeInTheDocument());
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("built-in rows have no delete button", async () => {
    await setup();
    const menu = await openMenu();
    await within(menu).findByText("Fast");
    expect(within(menu).queryByRole("button", { name: "Delete preset Fast" })).not.toBeInTheDocument();
    expect(within(menu).getByRole("button", { name: "Delete preset My mix" })).toBeInTheDocument();
  });

  it("the list shows an error line with a retry when listing fails", async () => {
    const presets = fakePresets();
    presets.list.mockRejectedValueOnce(new Error("boom")).mockRejectedValueOnce(new Error("boom"));
    await setup(presets);
    const menu = await openMenu();
    expect(await within(menu).findByText(/Could not list the presets: boom/)).toBeInTheDocument();
    await userEvent.click(within(menu).getByRole("button", { name: "Retry" }));
    expect(await within(menu).findByText("Fast")).toBeInTheDocument();
  });
});
