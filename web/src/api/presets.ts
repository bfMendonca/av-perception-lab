// Client of the preset routes (src/det_depth_distance/web/app.py): built-in and
// user presets are listed, loaded, saved and deleted by id.
import { readError } from "./client";
import type { ConfigDict, ParsedConfig } from "./types";

export interface PresetInfo {
  id: string;
  name: string;
  description: string;
  kind: "builtin" | "user";
}

export interface PresetsApi {
  list(): Promise<PresetInfo[]>;
  get(id: string): Promise<ParsedConfig>;
  /** Rejects with `ApiError` (status 409) when the name exists and `overwrite` is false, or clashes with a built-in. */
  save(name: string, config: ConfigDict, extras: ConfigDict, overwrite: boolean): Promise<PresetInfo>;
  remove(id: string): Promise<void>;
}

export function createPresetsApi(base = "", fetchImpl: typeof fetch = (...a) => fetch(...a)): PresetsApi {
  async function call(path: string, init?: RequestInit): Promise<Response> {
    const response = await fetchImpl(`${base}${path}`, init);
    if (!response.ok) throw await readError(response);
    return response;
  }
  const enc = encodeURIComponent;

  return {
    list: async () => (await call("/api/presets")).json() as Promise<PresetInfo[]>,
    get: async (id) => (await call(`/api/presets/${enc(id)}`)).json() as Promise<ParsedConfig>,
    save: async (name, config, extras, overwrite) =>
      (
        await call("/api/presets", {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name, config, extras, overwrite }),
        })
      ).json() as Promise<PresetInfo>,
    remove: async (id) => {
      await call(`/api/presets/${enc(id)}`, { method: "DELETE" });
    },
  };
}
