// Client of the preset routes (src/det_depth_distance/web/app.py): built-in and
// user presets are listed, loaded, saved and deleted by id.
import { ApiError, readError } from "./client";
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
  /** Rejects with `ApiError` (status 409) when the name exists and `overwrite` is false; 422 when it clashes with a built-in name. */
  save(name: string, config: ConfigDict, extras: ConfigDict, overwrite: boolean): Promise<PresetInfo>;
  remove(id: string): Promise<void>;
}

/** A validation 422 carries `[{path, message}]`: show it as one `path: message` line each. */
async function readPresetError(response: Response): Promise<ApiError> {
  try {
    const body = (await response.clone().json()) as { detail?: unknown };
    if (Array.isArray(body.detail) && body.detail.length > 0) {
      const lines = body.detail.map((d: { path?: unknown; message?: unknown }) =>
        d && typeof d === "object" && typeof d.message === "string" ? (typeof d.path === "string" && d.path ? `${d.path}: ${d.message}` : d.message) : JSON.stringify(d),
      );
      return new ApiError(lines.join("\n"), response.status);
    }
  } catch {
    // not JSON: readError keeps the status line
  }
  return readError(response);
}

export function createPresetsApi(base = "", fetchImpl: typeof fetch = (...a) => fetch(...a)): PresetsApi {
  async function call(path: string, init?: RequestInit): Promise<Response> {
    const response = await fetchImpl(`${base}${path}`, init);
    if (!response.ok) throw await readPresetError(response);
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
