import { describe, expect, it, vi } from "vitest";
import { ApiError } from "./client";
import { createPresetsApi } from "./presets";

function fakeFetch(status: number, body?: unknown) {
  return vi.fn(async (_url: string, _init?: RequestInit) =>
    new Response(body === undefined ? null : JSON.stringify(body), { status }),
  );
}
const asFetch = (f: unknown) => f as typeof fetch;

describe("presets api", () => {
  it("list GETs /api/presets", async () => {
    const f = fakeFetch(200, [{ id: "builtin:a", name: "a", description: "", kind: "builtin" }]);
    const rows = await createPresetsApi("", asFetch(f)).list();
    expect(f.mock.calls[0][0]).toBe("/api/presets");
    expect(rows[0].id).toBe("builtin:a");
  });

  it("get encodes the id", async () => {
    const f = fakeFetch(200, { config: {}, extras: {} });
    await createPresetsApi("", asFetch(f)).get("user:a b");
    expect(f.mock.calls[0][0]).toBe("/api/presets/user%3Aa%20b");
  });

  it("save PUTs the JSON body", async () => {
    const f = fakeFetch(200, { id: "user:x", name: "x", description: "", kind: "user" });
    const info = await createPresetsApi("", asFetch(f)).save("x", { a: 1 }, { b: 2 }, true);
    const [url, init] = f.mock.calls[0];
    expect(url).toBe("/api/presets");
    expect(init?.method).toBe("PUT");
    expect(JSON.parse(init?.body as string)).toEqual({ name: "x", config: { a: 1 }, extras: { b: 2 }, overwrite: true });
    expect(info.kind).toBe("user");
  });

  it("remove sends DELETE and accepts 204", async () => {
    const f = fakeFetch(204);
    await createPresetsApi("", asFetch(f)).remove("user:x");
    expect(f.mock.calls[0][1]?.method).toBe("DELETE");
  });

  it("a 409 rejects with ApiError status 409", async () => {
    const f = fakeFetch(409, { detail: "exists" });
    const err = await createPresetsApi("", asFetch(f)).save("x", {}, {}, false).catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(409);
    expect(err.message).toBe("exists");
  });

  it("a 422 with a validation list becomes path: message lines", async () => {
    const f = fakeFetch(422, {
      detail: [
        { path: "detectors[0].type", message: "unknown detector" },
        { path: "render.hud", message: "must be a boolean" },
      ],
    });
    const err = await createPresetsApi("", asFetch(f)).save("x", {}, {}, false).catch((e) => e);
    expect(err.status).toBe(422);
    expect(err.message).toBe("detectors[0].type: unknown detector\nrender.hud: must be a boolean");
  });
});
