import { act, render, renderHook, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { App } from "../App";
import type { ModelStatus, ModelsApi, StatusReport } from "../api/models";
import { StatusRejected } from "../api/models";
import { createEditorStore } from "../editor/yamlState";
import { fakeClient, fakeModelsApi, makeCatalog, modelRow, sampleConfig } from "../test/fixtures";
import { useModelStatus } from "./useModelStatus";

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
    expect(banner).toHaveTextContent("2 models need preparing · 1.5 GB");
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
    expect(await screen.findByRole("region", { name: "Model readiness" })).toHaveTextContent("1 model needs preparing · 8.0 MB");
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
