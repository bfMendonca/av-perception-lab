import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createClassesApi, resetClassesCache } from "../api/classes";
import type { Field } from "../api/types";
import { ClassSelect } from "./ClassSelect";
import { fieldControls } from "./fieldControls";
import { CONFIG_CONTEXT_KEY, StageContext } from "./registry";

const field: Field = { name: "semantic_classes", type: "any", nullable: true, default: null, required: false, variants: [{ type: "str" }, { type: "list", items: { type: "str" } }] };
const SCENE = { name: "scene", type: "segformer", model: "nvidia/segformer" };
const LANES = { name: "lanes", type: "twinlitenet" };

type Reply = { status: number; body: unknown };
/** A fetch answering by the spec's map name; records every call. */
function stubFetch(replies: Record<string, Reply>) {
  const f = vi.fn(async (_url: string, init?: RequestInit) => {
    const body = JSON.parse(String(init?.body)) as { spec: { name: string } };
    const reply = replies[body.spec.name];
    return new Response(JSON.stringify(reply.body), { status: reply.status });
  });
  vi.stubGlobal("fetch", f);
  return f;
}
const known = (names: string[]): Reply => ({ status: 200, body: { names, source: "static" } });

function renderSelect(value: unknown, specs: unknown[] = [SCENE]) {
  const onChange = vi.fn();
  const view = render(
    <StageContext.Provider value={{ [CONFIG_CONTEXT_KEY]: { semantic: specs } }}>
      <ClassSelect field={field} value={value} onChange={onChange} path={["render", "semantic_classes"]} errors={[]} id="cls" />
    </StageContext.Provider>,
  );
  return { onChange, ...view };
}

async function open() {
  await userEvent.click(screen.getByRole("button", { name: /classes/i }));
}

beforeEach(() => resetClassesCache());
afterEach(() => vi.unstubAllGlobals());

describe("ClassSelect", () => {
  it("test_all_checked_by_default", async () => {
    stubFetch({ scene: known(["road", "sky", "car"]) });
    renderSelect(undefined);
    await open();
    await screen.findByRole("checkbox", { name: "scene.road" });
    for (const name of ["scene.road", "scene.sky", "scene.car"]) expect(screen.getByRole("checkbox", { name })).toBeChecked();
    expect(screen.getByRole("group", { name: "scene" })).toBeInTheDocument();
  });

  it("test_unchecking_writes_allowlist", async () => {
    stubFetch({ scene: known(["road", "sky", "car"]) });
    const { onChange } = renderSelect(null);
    await open();
    await userEvent.click(await screen.findByRole("checkbox", { name: "scene.sky" }));
    expect(onChange).toHaveBeenLastCalledWith(["scene.road", "scene.car"]);
  });

  it("test_checking_everything_again_writes_undefined", async () => {
    stubFetch({ scene: known(["road", "sky", "car"]) });
    const { onChange } = renderSelect(["scene.road", "scene.car"]);
    await open();
    const sky = await screen.findByRole("checkbox", { name: "scene.sky" });
    expect(sky).not.toBeChecked();
    await userEvent.click(sky);
    expect(onChange).toHaveBeenLastCalledWith(undefined);
  });

  it("test_select_none_writes_empty_list", async () => {
    stubFetch({ scene: known(["road", "sky"]) });
    const { onChange } = renderSelect(undefined);
    await open();
    await screen.findByRole("checkbox", { name: "scene.road" });
    await userEvent.click(screen.getByRole("button", { name: /select none/i }));
    expect(onChange).toHaveBeenLastCalledWith([]);
    await userEvent.click(screen.getByRole("button", { name: /select all/i }));
    expect(onChange).toHaveBeenLastCalledWith(undefined);
  });

  it("test_search_filters_visible_entries", async () => {
    stubFetch({ scene: known(["road", "sky", "sidewalk"]) });
    renderSelect(undefined);
    await open();
    await screen.findByRole("checkbox", { name: "scene.road" });
    await userEvent.type(screen.getByRole("searchbox", { name: /filter classes/i }), "si");
    expect(screen.getByRole("checkbox", { name: "scene.sidewalk" })).toBeInTheDocument();
    expect(screen.queryByRole("checkbox", { name: "scene.road" })).not.toBeInTheDocument();
    expect(screen.queryByRole("checkbox", { name: "scene.sky" })).not.toBeInTheDocument();
  });

  it("test_background_is_disabled_with_note", async () => {
    stubFetch({ scene: known(["background", "road", "sky"]) });
    const { onChange } = renderSelect(null);
    await open();
    const bg = await screen.findByRole("checkbox", { name: "scene.background" });
    expect(bg).toBeDisabled();
    expect(bg).not.toBeChecked();
    expect(bg).toHaveAccessibleDescription(/never drawn/i);
    // background neither enters the allowlist nor stops "all" from being all
    await userEvent.click(screen.getByRole("checkbox", { name: "scene.road" }));
    expect(onChange).toHaveBeenLastCalledWith(["scene.sky"]);
  });

  it("test_unknown_names_shows_message", async () => {
    stubFetch({ scene: { status: 200, body: { names: null, source: null } }, lanes: { status: 422, body: { errors: [{ path: "spec.weights", message: "outside the roots" }] } } });
    renderSelect(undefined, [SCENE, LANES]);
    await open();
    await waitFor(() => expect(screen.getAllByText(/classes appear after the first load/i)).toHaveLength(2));
    expect(screen.queryByRole("checkbox")).not.toBeInTheDocument();
  });

  it("locks per-class editing of 'all' while another map's classes are unknown", async () => {
    stubFetch({ scene: known(["road", "sky"]), lanes: { status: 200, body: { names: null, source: null } } });
    const { onChange } = renderSelect(undefined, [SCENE, LANES]);
    await open();
    const road = await screen.findByRole("checkbox", { name: "scene.road" });
    await screen.findByText(/classes appear after the first load/i);
    expect(road).toBeChecked();
    expect(road).toBeDisabled();
    expect(screen.getByText(/per-class selection is locked/i)).toBeInTheDocument();
    await userEvent.click(road);
    expect(onChange).not.toHaveBeenCalled(); // lanes was never silently left out
    await userEvent.click(screen.getByRole("button", { name: /select none/i }));
    expect(onChange).toHaveBeenLastCalledWith([]);
  });

  it("locks while another map is still loading", async () => {
    stubFetch({ scene: known(["road"]), lanes: known(["lane"]) });
    resetClassesCache();
    renderSelect(undefined, [SCENE, LANES]);
    await open();
    // nothing fetched yet: both maps loading, the lock note explains why
    expect(screen.getAllByText(/loading classes/i)).toHaveLength(2);
    expect(screen.getByText(/per-class selection is locked/i)).toBeInTheDocument();
    expect(await screen.findByRole("checkbox", { name: "lanes.lane" })).toBeEnabled();
    expect(screen.queryByText(/per-class selection is locked/i)).not.toBeInTheDocument();
  });

  it("an explicit allowlist stays editable with an unknown map", async () => {
    stubFetch({ scene: known(["road", "sky"]), lanes: { status: 200, body: { names: null, source: null } } });
    const { onChange } = renderSelect(["scene.road", "scene.sky"], [SCENE, LANES]);
    await open();
    await screen.findByText(/classes appear after the first load/i);
    await waitFor(() => expect(screen.queryByText(/loading classes/i)).not.toBeInTheDocument());
    const road = screen.getByRole("checkbox", { name: "scene.road" });
    expect(road).toBeEnabled();
    await userEvent.click(road);
    expect(onChange).toHaveBeenLastCalledWith(["scene.sky"]);
  });

  it("keeps a YAML entry that is not in the fetched list", async () => {
    stubFetch({ scene: known(["road", "sky"]) });
    const { onChange } = renderSelect(["scene.road", "scene.lamp"]);
    await open();
    const kept = await screen.findByRole("checkbox", { name: "scene.lamp" });
    expect(kept).toBeChecked();
    await userEvent.click(screen.getByRole("checkbox", { name: "scene.road" }));
    expect(onChange).toHaveBeenLastCalledWith(["scene.lamp"]);
  });

  it("a bare string value counts as a one-item list", async () => {
    stubFetch({ scene: known(["road", "sky"]) });
    renderSelect("scene.sky");
    await open();
    expect(await screen.findByRole("checkbox", { name: "scene.road" })).not.toBeChecked();
    expect(screen.getByRole("checkbox", { name: "scene.sky" })).toBeChecked();
  });

  it("fetches each distinct spec once, across re-renders and remounts", async () => {
    const f = stubFetch({ scene: known(["road"]), lanes: known(["lane"]) });
    const { rerender, unmount } = renderSelect(undefined, [SCENE, LANES]);
    await open();
    await screen.findByRole("checkbox", { name: "lanes.lane" });
    const again = (value: unknown) => (
      <StageContext.Provider value={{ [CONFIG_CONTEXT_KEY]: { semantic: [{ ...SCENE }, { ...LANES }], render: { hud: true } } }}>
        <ClassSelect field={field} value={value} onChange={() => {}} path={["render", "semantic_classes"]} errors={[]} id="cls" />
      </StageContext.Provider>
    );
    rerender(again(["scene.road"]));
    unmount();
    render(again(undefined));
    await open();
    await screen.findByRole("checkbox", { name: "scene.road" });
    expect(f).toHaveBeenCalledTimes(2);
  });

  it("is the control of the semantic_classes field", async () => {
    stubFetch({ scene: known(["road"]) });
    render(
      <StageContext.Provider value={{ [CONFIG_CONTEXT_KEY]: { semantic: [SCENE] } }}>
        {fieldControls(field, { value: undefined, onChange: () => {}, path: ["render", "semantic_classes"] })}
      </StageContext.Provider>,
    );
    // the field label names the dropdown button
    expect(screen.getByRole("button", { name: /semantic.classes/i })).toHaveAttribute("aria-expanded", "false");
  });
});

describe("classes api", () => {
  it("POSTs kind and spec to /api/classes", async () => {
    const f = vi.fn(async () => new Response(JSON.stringify({ names: ["a"], source: "cache" }), { status: 200 }));
    const out = await createClassesApi("", f as unknown as typeof fetch).fetch("semantic", SCENE);
    const [url, init] = f.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe("/api/classes");
    expect(init.method).toBe("POST");
    expect(JSON.parse(String(init.body))).toEqual({ kind: "semantic", spec: SCENE });
    expect(out).toEqual({ names: ["a"], source: "cache" });
  });

  it("a rejected spec (422) means no list", async () => {
    const f = vi.fn(async () => new Response(JSON.stringify({ errors: [{ path: "spec.type", message: "unknown" }] }), { status: 422 }));
    expect(await createClassesApi("", f as unknown as typeof fetch).fetch("semantic", {})).toEqual({ names: null, source: null });
  });
});
