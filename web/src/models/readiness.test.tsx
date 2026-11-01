import { act, render, renderHook, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { App } from "../App";
import type { ModelStatus, ModelsApi, StatusReport } from "../api/models";
import { StatusRejected } from "../api/models";
import { createEditorStore } from "../editor/yamlState";
import { fakeClient, fakeModelsApi, makeCatalog, modelRow, sampleConfig } from "../test/fixtures";
import type { PrepareJob } from "../api/models";
import { rowMatchesConfig, useModelStatus } from "./useModelStatus";

const ready = (patch: Partial<ModelStatus> = {}) => modelRow({ state: "ready", layers: { assets: "ready", code: "ready", packages: "ready" }, missing: [], size_bytes: null, message: null, ...patch });

async function renderEditor(rows: ModelStatus[], config = sampleConfig.config) {
  const client = fakeClient(makeCatalog(), { config, extras: {} });
  const store = createEditorStore(client, { debounceMs: 0 });
  await store.load("ignored", "pipeline.yaml");
  const models = fakeModelsApi(rows);
  render(<App client={client} store={store} modelsApi={models} />);
  const lane = await screen.findByRole("region", { name: "Detectors" });
  return { models, store, lane };
}

const cardOf = (lane: HTMLElement, name: RegExp) => within(lane).getByRole("button", { name });

describe("model readiness on the cards", () => {
  it("missing card renders dashed warning style and an alert icon", async () => {
    const { lane } = await renderEditor([modelRow()]);
    const card = cardOf(lane, /alpha/i);
    await waitFor(() => expect(card).toHaveClass("card--notready"));
    expect(card).toHaveAttribute("data-model-state", "missing");
    const alert = within(card).getByRole("img", { name: /not ready/i });
    expect(alert).toHaveClass("card__alert");
    expect(alert).toHaveAccessibleName(/<weights root>\/alpha\/large\.bin not found/);
    expect(card).toHaveTextContent("8.0 MB to download");
  });

  it("ready card has neither", async () => {
    const { lane, models } = await renderEditor([ready()]);
    const card = cardOf(lane, /alpha/i);
    await waitFor(() => expect(card).toHaveAttribute("data-model-state", "ready"));
    expect(models.statusCalls.length).toBeGreaterThan(0);
    expect(card).toHaveClass("card--ready");
    expect(card).not.toHaveClass("card--notready");
    expect(card).not.toHaveClass("card--failed");
    expect(within(card).queryByRole("img")).not.toBeInTheDocument();
    expect(within(card).queryByRole("progressbar")).not.toBeInTheDocument();
    // a module without a status row is a plain card
    const aggregation = await screen.findByRole("region", { name: "Aggregation" });
    expect(within(aggregation).getByRole("button", { name: /merge/i })).not.toHaveClass("card--model");
  });

  it("unknown state value renders the generic style", async () => {
    const { lane } = await renderEditor([modelRow({ state: "quarantined", message: "held for review" })]);
    const card = cardOf(lane, /alpha/i);
    await waitFor(() => expect(card).toHaveAttribute("data-model-state", "quarantined"));
    expect(card).toHaveClass("card--notready");
    expect(within(card).getByRole("img", { name: /quarantined: held for review/i })).toBeInTheDocument();
    expect(card).toHaveTextContent("Quarantined");
  });

  it("conflict renders as a warning with its message, and no Prepare action", async () => {
    const { lane } = await renderEditor([modelRow({ state: "conflict", message: "the resolver found no solution" })]);
    const card = cardOf(lane, /alpha/i);
    await waitFor(() => expect(card).toHaveClass("card--notready"));
    expect(within(card).getByRole("img", { name: /conflict: the resolver/i })).toBeInTheDocument();
    await userEvent.click(card);
    const section = await screen.findByRole("region", { name: "Weights & requirements" });
    expect(within(section).getByText("the resolver found no solution")).toBeInTheDocument();
    expect(within(section).queryByRole("button", { name: /prepare model/i })).not.toBeInTheDocument();
  });

  it("preparing card shows progress from SSE events", async () => {
    const { lane, models } = await renderEditor([modelRow()]);
    const banner = await screen.findByRole("region", { name: "Model readiness" });
    await userEvent.click(within(banner).getByRole("button", { name: "Prepare all" }));
    await waitFor(() => expect(models.prepared).toHaveLength(1));
    const card = cardOf(lane, /alpha/i);
    await waitFor(() => expect(card).toHaveClass("card--preparing"));
    act(() => {
      models.emit("prep1", { event: "running" });
      models.emit("prep1", { event: "model", ...modelRow({ state: "preparing" }), item: null, done: 0, total: null });
      models.emit("prep1", { event: "progress", stage: "detectors", index: 0, item: "weights", done: 2 * 1024 * 1024, total: 8 * 1024 * 1024 });
    });
    const bar = within(card).getByRole("progressbar", { name: "Preparing alpha" });
    expect(bar).toHaveAttribute("aria-valuenow", "25");
    expect(bar).toHaveAttribute("aria-valuetext", "2.0 MB of 8.0 MB");
    expect(within(card).queryByRole("img")).not.toBeInTheDocument(); // preparing: bar, not alert
    act(() => models.emit("prep1", { event: "progress", stage: "detectors", index: 0, item: "weights", done: 6 * 1024 * 1024, total: 8 * 1024 * 1024 }));
    expect(bar).toHaveAttribute("aria-valuenow", "75");
    // the banner follows the overall progress and offers Cancel
    const status = screen.getByRole("status", { name: "Model readiness" });
    expect(within(status).getByRole("progressbar", { name: "Prepare progress" })).toHaveAttribute("aria-valuenow", "75");
    await userEvent.click(within(status).getByRole("button", { name: "Cancel" }));
    expect(models.cancelled).toEqual(["prep1"]);

    // done: the status is checked again and the card turns solid
    models.rows = [ready()];
    act(() => {
      models.emit("prep1", { event: "model", ...ready(), item: "weights", done: 8, total: 8 });
      models.emit("prep1", { event: "end", status: "done", error: null });
    });
    await waitFor(() => expect(card).toHaveAttribute("data-model-state", "ready"));
    expect(card).not.toHaveClass("card--preparing");
    expect(screen.queryByRole("region", { name: "Model readiness" })).not.toBeInTheDocument();
  });

  it("a failed prepare stays red dashed with its reason and offers Retry", async () => {
    const { lane, models } = await renderEditor([modelRow()]);
    await userEvent.click(within(await screen.findByRole("region", { name: "Model readiness" })).getByRole("button", { name: "Prepare all" }));
    await waitFor(() => expect(models.prepared).toHaveLength(1));
    act(() => {
      models.emit("prep1", { event: "model", ...modelRow({ state: "failed", message: "offline or unreachable: example.invalid" }), item: null, done: 0, total: null });
      models.emit("prep1", { event: "end", status: "failed", error: "detectors[0] alpha: offline or unreachable: example.invalid" });
    });
    const card = cardOf(lane, /alpha/i);
    // the refreshed status says "missing" again; the attempt keeps it failed
    await waitFor(() => expect(models.statusCalls.length).toBeGreaterThan(1));
    await waitFor(() => expect(card).toHaveClass("card--failed"));
    expect(within(card).getByRole("img", { name: /failed: offline or unreachable/i })).toBeInTheDocument();
    await userEvent.click(card);
    const section = await screen.findByRole("region", { name: "Weights & requirements" });
    expect(within(section).getByRole("alert")).toHaveTextContent("offline or unreachable: example.invalid");
    await userEvent.click(within(section).getByRole("button", { name: /retry/i }));
    await waitFor(() => expect(models.prepared).toHaveLength(2));
    expect(models.prepared[1].models).toEqual([{ stage: "detectors", index: 0 }]);
  });
});

describe("readiness banner", () => {
  it("banner shows count and total size and triggers prepare all", async () => {
    const rows = [modelRow(), modelRow({ stage: "camera.estimate", index: null, type: "beta", name: "focal", size_bytes: 1.5 * 1024 ** 3, message: null }), ready({ stage: "aggregation", index: null, type: "merge", name: "merge" })];
    const { models } = await renderEditor(rows);
    const banner = await screen.findByRole("region", { name: "Model readiness" });
    expect(banner).toHaveTextContent("2 models need preparing · 1.5 GB to download");
    // rows without a lane card are listed too
    await userEvent.click(within(banner).getByRole("button", { name: "Show models" }));
    const list = within(banner).getByRole("list", { name: "Models that are not ready" });
    expect(within(list).getAllByRole("listitem")).toHaveLength(2);
    expect(list).toHaveTextContent("camera.estimate");
    await userEvent.click(within(banner).getByRole("button", { name: "Prepare all" }));
    await waitFor(() => expect(models.prepared).toHaveLength(1));
    expect(models.prepared[0].models).toBe("all");
    expect(models.prepared[0].config.detectors).toEqual(sampleConfig.config.detectors);
  });

  it("is hidden when every model is ready, and says so when the status cannot be checked", async () => {
    const { models } = await renderEditor([ready()]);
    await waitFor(() => expect(models.statusCalls.length).toBeGreaterThan(0));
    await waitFor(() => expect(screen.getByRole("button", { name: /alpha/i })).toHaveAttribute("data-model-state", "ready"));
    expect(screen.queryByRole("region", { name: "Model readiness" })).not.toBeInTheDocument();
    expect(screen.queryByRole("status", { name: "Model readiness" })).not.toBeInTheDocument();
  });

  it("offers a retry when the status is unavailable", async () => {
    const client = fakeClient();
    const store = createEditorStore(client, { debounceMs: 0 });
    await store.load("ignored", "p.yaml");
    const models = fakeModelsApi([modelRow()]);
    models.failWith = new Error("502 Bad Gateway");
    render(<App client={client} store={store} modelsApi={models} />);
    const status = await screen.findByRole("status", { name: "Model readiness" });
    expect(status).toHaveTextContent("Could not check whether the models are ready");
    expect(status).toHaveTextContent("502 Bad Gateway");
    models.failWith = null;
    await userEvent.click(within(status).getByRole("button", { name: "Check again" }));
    expect(await screen.findByRole("region", { name: "Model readiness" })).toHaveTextContent("1 model needs preparing · 8.0 MB to download");
  });
});

describe("inspector: weights & requirements", () => {
  it("inspector shows Prepare model for a missing model and nothing for a model with no requirements", async () => {
    const { lane, models } = await renderEditor([modelRow()]);
    await userEvent.click(cardOf(lane, /alpha/i));
    const section = await screen.findByRole("region", { name: "Weights & requirements" });
    expect(section).toHaveTextContent("Not ready");
    expect(section).toHaveTextContent("8.0 MB to download");
    expect(section).toHaveTextContent("<weights root>/alpha/large.bin");
    expect(within(section).getByText("weights", { selector: "code.reqs__missing" })).toBeInTheDocument();
    // the module's path field is the override
    expect(section).toHaveTextContent(/Using your own file\? Set weights under Options to override/);
    await userEvent.click(within(section).getByRole("button", { name: /prepare model/i }));
    await waitFor(() => expect(models.prepared).toHaveLength(1));
    expect(models.prepared[0].models).toEqual([{ stage: "detectors", index: 0 }]);

    // a module without a status row (no requirements) has no section
    await userEvent.click(within(await screen.findByRole("region", { name: "Aggregation" })).getByRole("button", { name: /merge/i }));
    await screen.findByRole("heading", { name: "merge" });
    expect(screen.queryByRole("region", { name: "Weights & requirements" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /prepare model/i })).not.toBeInTheDocument();
  });

  it("a ready model shows its state but no Prepare action", async () => {
    const { lane } = await renderEditor([ready()]);
    await waitFor(() => expect(cardOf(lane, /alpha/i)).toHaveAttribute("data-model-state", "ready"));
    await userEvent.click(cardOf(lane, /alpha/i));
    const section = await screen.findByRole("region", { name: "Weights & requirements" });
    expect(section).toHaveTextContent("nothing to prepare");
    expect(within(section).queryByRole("button", { name: /prepare model|retry/i })).not.toBeInTheDocument();
  });

  it("an unknown source shows the CLI command instead of a Prepare button", async () => {
    const { lane, models } = await renderEditor([modelRow({ message: "unknown source: prepare it from the CLI (--prepare)" })]);
    await userEvent.click(cardOf(lane, /alpha/i));
    const section = await screen.findByRole("region", { name: "Weights & requirements" });
    expect(section).toHaveTextContent("unknown source: prepare it from the CLI (--prepare)");
    expect(within(section).queryByRole("button", { name: /prepare model/i })).not.toBeInTheDocument();
    expect(within(section).getByLabelText("Command")).toHaveTextContent("det-depth-distance --config pipeline.yaml --prepare");
    expect(within(section).getByRole("button", { name: "Copy command" })).toBeInTheDocument();
    // and Prepare all has nothing it may do
    expect(within(screen.getByRole("region", { name: "Model readiness" })).getByRole("button", { name: "Prepare all" })).toBeDisabled();
    expect(models.prepared).toHaveLength(0);
  });

  it("rows below a general section show in its settings tab", async () => {
    const { store } = await renderEditor([modelRow({ stage: "render.overlay", index: null, type: "beta", name: "overlay" })]);
    act(() => store.select({ kind: "general", stageKey: "render" }));
    const section = await screen.findByRole("region", { name: "Weights & requirements" });
    expect(section).toHaveTextContent("render.overlay");
    expect(within(section).getByRole("button", { name: /prepare model/i })).toBeInTheDocument();
  });
});

describe("useModelStatus", () => {
  it("drops a stale answer when the config changed meanwhile", async () => {
    const answers: ((r: StatusReport) => void)[] = [];
    const api = {
      ...fakeModelsApi(),
      status: () => new Promise<StatusReport>((resolve) => answers.push(resolve)),
    } as ModelsApi;
    const { result, rerender } = renderHook(({ config }) => useModelStatus(config, { api, debounceMs: 0 }), { initialProps: { config: { a: 1 } as Record<string, unknown> } });
    await waitFor(() => expect(answers).toHaveLength(1));
    rerender({ config: { a: 2 } });
    await waitFor(() => expect(answers).toHaveLength(2));
    act(() => answers[1]({ models: [ready()], total_size_bytes: 0 }));
    await waitFor(() => expect(result.current.status).toBe("ready"));
    act(() => answers[0]({ models: [modelRow()], total_size_bytes: 8 })); // late answer to {a: 1}
    expect(result.current.rows.map((r) => r.state)).toEqual(["ready"]);
    expect(result.current.notReady).toHaveLength(0);
  });

  it("an invalid config clears the rows and blocks nothing", async () => {
    const api = { ...fakeModelsApi(), status: async () => Promise.reject(new StatusRejected([{ path: "depth", message: "required" }])) } as ModelsApi;
    const { result } = renderHook(() => useModelStatus({ a: 1 }, { api, debounceMs: 0 }));
    await waitFor(() => expect(result.current.status).toBe("invalid"));
    expect(result.current.rows).toEqual([]);
  });
});

// -- fix round 1: lifecycle, identity, server fields ------------------------------

const MB = 1024 * 1024;
const trio = {
  ...sampleConfig.config,
  detectors: [
    { type: "alpha", name: "first" },
    { type: "beta", name: "second" },
    { type: "alpha", name: "third" },
  ],
};
const rowOf = (index: number, name: string, type: string, patch: Partial<ModelStatus> = {}) => modelRow({ index, name, type, message: `weights: <weights root>/${name}.bin not found`, ...patch });
const event = (index: number, name: string, type: string, state: string) => ({ event: "model", ...rowOf(index, name, type, { state }), item: null, done: 0, total: null });
const progress = (index: number, done: number, total = 8 * MB) => ({ event: "progress", stage: "detectors", index, item: "weights", done, total });

describe("prepare lifecycle (fix round)", () => {
  it("a model that finished stays ready while the rest of the job runs", async () => {
    const { lane, models } = await renderEditor([rowOf(0, "first", "alpha"), rowOf(1, "second", "beta"), rowOf(2, "third", "alpha")], trio);
    await userEvent.click(within(await screen.findByRole("region", { name: "Model readiness" })).getByRole("button", { name: "Prepare all" }));
    await waitFor(() => expect(models.prepared).toHaveLength(1));
    act(() => {
      models.emit("prep1", event(0, "first", "alpha", "preparing"));
      models.emit("prep1", progress(0, 8 * MB));
      models.emit("prep1", event(0, "first", "alpha", "ready"));
      models.emit("prep1", event(1, "second", "beta", "preparing"));
    });
    const first = cardOf(lane, /first/);
    // the server row still says "missing" (no status asked since): no flash back
    expect(first).toHaveAttribute("data-model-state", "ready");
    expect(first).not.toHaveClass("card--notready");
    expect(cardOf(lane, /second/)).toHaveClass("card--preparing");
    await userEvent.click(first);
    const section = await screen.findByRole("region", { name: "Weights & requirements" });
    expect(within(section).queryByRole("button", { name: /prepare model|retry/i })).not.toBeInTheDocument();
    // job ends; the next answer is the server's truth
    models.rows = [rowOf(0, "first", "alpha", { state: "ready", missing: [], message: null }), rowOf(1, "second", "beta", { state: "ready", missing: [], message: null }), rowOf(2, "third", "alpha")];
    const calls = models.statusCalls.length;
    act(() => {
      models.emit("prep1", event(1, "second", "beta", "ready"));
      models.emit("prep1", { event: "end", status: "done", error: null });
    });
    await waitFor(() => expect(models.statusCalls.length).toBeGreaterThan(calls));
    await waitFor(() => expect(cardOf(lane, /second/)).toHaveAttribute("data-model-state", "ready"));
    expect(first).toHaveAttribute("data-model-state", "ready");
  });

  it("re-attaches to a job started before a reload: progress and Cancel", async () => {
    const running: PrepareJob = {
      id: "older",
      status: "running",
      position: null,
      error: null,
      created: "",
      cancel_requested: false,
      models: [
        { ...modelRow({ state: "preparing" }), item: "weights", done: 0, total: 8 * MB },
        { ...modelRow({ stage: "aggregation", index: null, type: "merge", name: "merge", state: "queued" }), item: null, done: 0, total: null },
      ],
      skipped: [],
    };
    const client = fakeClient();
    const store = createEditorStore(client, { debounceMs: 0 });
    await store.load("ignored", "p.yaml");
    const models = fakeModelsApi([modelRow({ state: "preparing" }), modelRow({ stage: "aggregation", index: null, type: "merge", name: "merge", state: "ready", missing: [], message: null })]);
    models.listed = [running, { ...running, id: "gone", status: "done" }];
    render(<App client={client} store={store} modelsApi={models} />);
    const lane = await screen.findByRole("region", { name: "Detectors" });
    act(() => models.emit("older", { event: "progress", stage: "detectors", index: 0, item: "weights", done: 4 * MB, total: 8 * MB }));
    expect(within(cardOf(lane, /alpha/)).getByRole("progressbar")).toHaveAttribute("aria-valuenow", "50");
    // queued in that job but ready on the server (attached before the rows came): ready
    const aggregation = screen.getByRole("region", { name: "Aggregation" });
    await waitFor(() => expect(within(aggregation).getByRole("button", { name: /merge/i })).toHaveAttribute("data-model-state", "ready"));
    act(() => models.emit("gone", { event: "progress", stage: "detectors", index: 0, item: "weights", done: 1, total: 8 * MB })); // finished jobs are not followed
    const status = screen.getByRole("status", { name: "Model readiness" });
    await userEvent.click(within(status).getByRole("button", { name: "Cancel" }));
    expect(models.cancelled).toEqual(["older"]);
  });

  it("a replayed stream never rewinds progress nor takes a finished model back", async () => {
    const { lane, models } = await renderEditor([rowOf(0, "first", "alpha"), rowOf(1, "second", "beta"), rowOf(2, "third", "alpha")], trio);
    await userEvent.click(within(await screen.findByRole("region", { name: "Model readiness" })).getByRole("button", { name: "Prepare all" }));
    await waitFor(() => expect(models.prepared).toHaveLength(1));
    act(() => {
      models.emit("prep1", { event: "running" });
      models.emit("prep1", event(0, "first", "alpha", "preparing"));
      models.emit("prep1", progress(0, 8 * MB));
      models.emit("prep1", event(0, "first", "alpha", "ready"));
      models.emit("prep1", event(1, "second", "beta", "preparing"));
      models.emit("prep1", progress(1, 6 * MB));
    });
    const second = within(cardOf(lane, /second/)).getByRole("progressbar");
    expect(second).toHaveAttribute("aria-valuenow", "75");
    // EventSource reconnect: the server replays the job from the start
    act(() => {
      models.emit("prep1", { event: "queued" });
      models.emit("prep1", { event: "running" });
      models.emit("prep1", event(0, "first", "alpha", "preparing"));
      models.emit("prep1", progress(0, 2 * MB));
      models.emit("prep1", event(1, "second", "beta", "queued"));
      models.emit("prep1", progress(1, 2 * MB));
    });
    expect(cardOf(lane, /first/)).toHaveAttribute("data-model-state", "ready");
    expect(within(cardOf(lane, /first/)).queryByRole("progressbar")).not.toBeInTheDocument();
    expect(cardOf(lane, /second/)).toHaveAttribute("data-model-state", "preparing");
    expect(within(cardOf(lane, /second/)).getByRole("progressbar")).toHaveAttribute("aria-valuenow", "75");
  });

  it("removing a module during a job never moves its progress to another card", async () => {
    const { lane, models, store } = await renderEditor([rowOf(0, "first", "alpha"), rowOf(1, "second", "beta"), rowOf(2, "third", "alpha")], trio);
    await userEvent.click(within(await screen.findByRole("region", { name: "Model readiness" })).getByRole("button", { name: "Prepare all" }));
    await waitFor(() => expect(models.prepared).toHaveLength(1));
    act(() => {
      models.emit("prep1", event(1, "second", "beta", "preparing"));
      models.emit("prep1", progress(1, 4 * MB));
    });
    expect(within(cardOf(lane, /second/)).getByRole("progressbar")).toHaveAttribute("aria-valuenow", "50");
    // remove "first": "second" moves to index 0, "third" to index 1
    models.rows = [rowOf(0, "second", "beta", { state: "preparing" }), rowOf(1, "third", "alpha")];
    act(() => {
      store.removeModule("detectors", 0);
    });
    // at once (old rows, new config): no card shows another module's progress
    expect(within(cardOf(lane, /third/)).queryByRole("progressbar")).not.toBeInTheDocument();
    expect(cardOf(lane, /third/)).not.toHaveAttribute("data-model-state");
    // after the new answer each card shows its own module: the progress follows "second"
    await waitFor(() => expect(within(cardOf(lane, /second/)).getByRole("progressbar")).toHaveAttribute("aria-valuenow", "50"));
    expect(cardOf(lane, /third/)).toHaveAttribute("data-model-state", "queued"); // its own overlay (queued in the job)
    expect(within(cardOf(lane, /third/)).getByRole("progressbar")).not.toHaveAttribute("aria-valuenow");
    act(() => models.emit("prep1", progress(1, 6 * MB))); // the job still names "second" by its old index
    expect(within(cardOf(lane, /second/)).getByRole("progressbar")).toHaveAttribute("aria-valuenow", "75");
    expect(within(cardOf(lane, /third/)).getByRole("progressbar")).not.toHaveAttribute("aria-valuenow");
  });

  it("reordering modules during a job keeps each state on its own card", async () => {
    const pair = { ...sampleConfig.config, detectors: [{ type: "alpha", name: "first" }, { type: "beta", name: "second" }] };
    const { lane, models, store } = await renderEditor([rowOf(0, "first", "alpha"), rowOf(1, "second", "beta", { state: "ready", missing: [], message: null })], pair);
    await userEvent.click(within(await screen.findByRole("region", { name: "Model readiness" })).getByRole("button", { name: "Prepare all" }));
    await waitFor(() => expect(models.prepared).toHaveLength(1));
    act(() => {
      models.emit("prep1", event(0, "first", "alpha", "preparing"));
      models.emit("prep1", progress(0, 4 * MB));
    });
    models.rows = [rowOf(0, "second", "beta", { state: "ready", missing: [], message: null }), rowOf(1, "first", "alpha", { state: "preparing" })];
    act(() => store.setOption(["detectors"], [pair.detectors[1], pair.detectors[0]]));
    expect(within(cardOf(lane, /second/)).queryByRole("progressbar")).not.toBeInTheDocument();
    await waitFor(() => expect(cardOf(lane, /second/)).toHaveAttribute("data-model-state", "ready"));
    expect(cardOf(lane, /first/)).toHaveAttribute("data-model-state", "preparing");
    expect(within(cardOf(lane, /first/)).getByRole("progressbar")).toHaveAttribute("aria-valuenow", "50"); // its progress moved with it
  });

  it("rowMatchesConfig checks type, name and index against the edited config", () => {
    const row = rowOf(1, "second", "beta");
    expect(rowMatchesConfig(row, trio)).toBe(true);
    expect(rowMatchesConfig(row, { detectors: [trio.detectors[0], trio.detectors[2]] })).toBe(false); // another module there now
    expect(rowMatchesConfig(row, { detectors: [trio.detectors[0]] })).toBe(false); // index past the end
    expect(rowMatchesConfig(row, { detectors: [{}, { type: "beta" }] })).toBe(true); // unnamed spec: server default name
    expect(rowMatchesConfig(modelRow({ stage: "render.overlay", index: null }), trio)).toBe(true); // sub-path: kept
    expect(rowMatchesConfig(row, {})).toBe(true); // key not set: server defaults
  });
});

describe("server display fields (fix round)", () => {
  it("renders source, licence, target and the server's action fields", async () => {
    const row = modelRow({ source: "Hugging Face · org/model; git · https://example.invalid/tool.git", license: "Apache-2.0", target: "hf; _code/tool", reason: null, preparable: true, message: "files missing" });
    const { lane } = await renderEditor([row]);
    const banner = await screen.findByRole("region", { name: "Model readiness" });
    expect(banner).toHaveTextContent("Licences: Apache-2.0.");
    expect(within(banner).getByRole("button", { name: "Prepare all" })).toHaveAttribute("title", expect.stringMatching(/8\.0 MB to download.*licences: Apache-2\.0/));
    await userEvent.click(cardOf(lane, /alpha/i));
    const section = await screen.findByRole("region", { name: "Weights & requirements" });
    expect(section).toHaveTextContent("Hugging Face · org/model");
    expect(section).toHaveTextContent("git · https://example.invalid/tool.git");
    expect(section).toHaveTextContent("Apache-2.0");
    expect(within(section).getByText("<weights root>/hf")).toBeInTheDocument();
    expect(within(section).getByText("<weights root>/_code/tool")).toBeInTheDocument();
    expect(within(section).getByRole("button", { name: /prepare model/i })).toBeInTheDocument();
  });

  it("the server's reason wins over the message wording", async () => {
    const { lane } = await renderEditor([modelRow({ reason: "unknown_source", preparable: false, message: "this source is not vouched for" })]);
    await userEvent.click(cardOf(lane, /alpha/i));
    const section = await screen.findByRole("region", { name: "Weights & requirements" });
    expect(within(section).queryByRole("button", { name: /prepare model/i })).not.toBeInTheDocument();
    expect(within(section).getByLabelText("Command")).toHaveTextContent("--prepare");
    const banner = screen.getByRole("region", { name: "Model readiness" });
    expect(banner).toHaveTextContent("1 model is not ready");
    expect(banner).toHaveTextContent("1 model cannot be prepared from here");
    expect(within(banner).getByRole("button", { name: "Prepare all" })).toBeDisabled();
  });

  it("an unknown-source message no longer hides Prepare when the server says preparable", async () => {
    const { lane } = await renderEditor([modelRow({ reason: null, preparable: true, message: "unknown source: wording changed" })]);
    await userEvent.click(cardOf(lane, /alpha/i));
    const section = await screen.findByRole("region", { name: "Weights & requirements" });
    expect(within(section).getByRole("button", { name: /prepare model/i })).toBeInTheDocument();
  });

  it("packages-only rows (server reason) have no Prepare and a note", async () => {
    const { lane } = await renderEditor([modelRow({ reason: "packages_only", preparable: false, missing: ["package:somepkg"], message: "package somepkg is not installed" })]);
    await userEvent.click(cardOf(lane, /alpha/i));
    const section = await screen.findByRole("region", { name: "Weights & requirements" });
    expect(within(section).queryByRole("button", { name: /prepare model/i })).not.toBeInTheDocument();
    expect(section).toHaveTextContent("Only Python packages are missing");
  });
});

describe("packages layer (task_0024)", () => {
  const INSTALL = "uv --color never sync --locked --inexact --extra infer --extra model-xyz --extra web";
  const pkgRow = (patch: Partial<ModelStatus> = {}) =>
    modelRow({
      layers: { assets: "ready", code: "ready", packages: "missing" },
      missing: ["package:xyzlib"],
      size_bytes: null,
      message: "package xyzlib is not installed (extra 'model-xyz')",
      reason: "packages_only",
      preparable: false,
      packages: { action: "install", extra: "model-xyz", command: INSTALL, message: "would install xyzlib==1.0" },
      ...patch,
    });

  async function openSection(rows: ModelStatus[]) {
    const editor = await renderEditor(rows);
    await userEvent.click(cardOf(editor.lane, /alpha/i));
    const section = await screen.findByRole("region", { name: "Weights & requirements" });
    return { ...editor, section };
  }

  it("Install packages asks for confirmation stating the command, and only Install starts the job", async () => {
    const { section, models } = await openSection([pkgRow()]);
    expect(within(section).queryByRole("button", { name: /prepare model/i })).not.toBeInTheDocument();
    expect(section).toHaveTextContent("A dry-run against the server's environment is clean (would install xyzlib==1.0)");
    await userEvent.click(within(section).getByRole("button", { name: /install packages/i }));
    let dialog = await screen.findByRole("dialog", { name: "Install packages" });
    expect(dialog).toHaveTextContent(INSTALL);
    expect(dialog).toHaveTextContent("nothing is removed");
    expect(dialog).toHaveTextContent("Restart the server afterwards");
    await userEvent.click(within(dialog).getByRole("button", { name: "Cancel" }));
    expect(models.prepared).toHaveLength(0); // cancelled: nothing runs
    await userEvent.click(within(section).getByRole("button", { name: /install packages/i }));
    dialog = await screen.findByRole("dialog", { name: "Install packages" });
    await userEvent.click(within(dialog).getByRole("button", { name: "Install" }));
    await waitFor(() => expect(models.prepared).toHaveLength(1));
    expect(models.prepared[0]).toMatchObject({ models: [{ stage: "detectors", index: 0 }], packages: true });
  });

  it("an install job shows uv output, then asks for a server restart", async () => {
    const { section, models } = await openSection([pkgRow()]);
    await userEvent.click(within(section).getByRole("button", { name: /install packages/i }));
    await userEvent.click(within(await screen.findByRole("dialog", { name: "Install packages" })).getByRole("button", { name: "Install" }));
    await waitFor(() => expect(models.prepared).toHaveLength(1));
    act(() => {
      models.emit("prep1", { event: "running" });
      models.emit("prep1", { event: "model", ...pkgRow({ state: "preparing" }), item: null, done: 0, total: null });
      models.emit("prep1", { event: "output", stage: "detectors", index: 0, line: `$ ${INSTALL}` });
      models.emit("prep1", { event: "output", stage: "detectors", index: 0, line: " + xyzlib==1.0" });
    });
    const live = screen.getByRole("region", { name: "Weights & requirements" });
    expect(live).toHaveTextContent("Installing packages…");
    expect(within(live).getByLabelText("uv output")).toHaveTextContent("+ xyzlib==1.0");
    models.rows = [ready()];
    models.restartRequired = true;
    act(() => {
      models.emit("prep1", { event: "model", ...ready(), installed: true, message: "packages installed: restart the server to use them", item: null, done: 0, total: null });
      models.emit("prep1", { event: "end", status: "done", error: null });
    });
    await waitFor(() => expect(screen.getByRole("status", { name: "Restart required" })).toHaveTextContent("Restart the server to use them"));
  });

  it("a missing base stack shows the exact one-time command with a copy button and no action button", async () => {
    const command = "uv sync --extra infer --extra web";
    const { section, models } = await openSection([
      pkgRow({ message: "package torch is not installed (extra 'infer')", missing: ["package:torch"], packages: { action: "base", extra: "infer", command, message: "the base Python stack is not installed" } }),
    ]);
    expect(within(section).getByLabelText("Command")).toHaveTextContent(command);
    expect(within(section).getByRole("button", { name: "Copy command" })).toBeInTheDocument();
    expect(section).toHaveTextContent("the server never installs it");
    expect(within(section).queryByRole("button", { name: /install packages/i })).not.toBeInTheDocument();
    expect(within(section).queryByRole("button", { name: /prepare model/i })).not.toBeInTheDocument();
    expect(models.prepared).toHaveLength(0);
  });

  it("a dry-run that could not run (uv unavailable) is failed with its reason, not a conflict", async () => {
    const reason = "cannot check extra 'model-xyz': uv is not available (No such file or directory): install uv or set DDD_UV";
    const { section } = await openSection([pkgRow({ state: "failed", message: reason, packages: { action: "error", extra: "model-xyz", command: null, message: reason } })]);
    expect(section).toHaveTextContent("set DDD_UV");
    expect(section).not.toHaveTextContent("conflicts with the current environment");
    expect(within(section).queryByRole("button", { name: /install packages/i })).not.toBeInTheDocument();
    expect(within(section).queryByRole("button", { name: /retry|prepare model/i })).not.toBeInTheDocument();
  });

  it("a conflict shows the resolver's reason and the docs pointer, with no install action", async () => {
    const reason = "installing extra 'model-xyz' conflicts with the current environment: × No solution found (see docs/model_environments.md)";
    const { section, lane } = await openSection([pkgRow({ state: "conflict", message: reason, packages: { action: "conflict", extra: "model-xyz", command: null, message: reason } })]);
    expect(cardOf(lane, /alpha/i)).toHaveClass("card--notready");
    expect(section).toHaveTextContent("No solution found");
    expect(within(section).getAllByText("docs/model_environments.md").length).toBeGreaterThan(0);
    expect(within(section).queryByRole("button", { name: /install packages/i })).not.toBeInTheDocument();
    expect(within(section).queryByRole("button", { name: /prepare model/i })).not.toBeInTheDocument();
  });
});
