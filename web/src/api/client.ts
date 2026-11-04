// Thin client of the config routes; injected so tests (and future views) can fake it.
import type { Catalog, ConfigDict, ParsedConfig, ValidationError } from "./types";

export interface ApiClient {
  catalog(): Promise<Catalog>;
  validate(config: ConfigDict): Promise<ValidationError[]>;
  parse(yaml: string): Promise<ParsedConfig>;
  dump(config: ConfigDict, extras: ConfigDict): Promise<string>;
}

/** An HTTP error with the server's `detail` message when there is one. */
export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export async function readError(response: Response): Promise<ApiError> {
  let message = `${response.status} ${response.statusText}`.trim();
  try {
    const body = (await response.json()) as { detail?: unknown };
    if (typeof body.detail === "string") message = body.detail;
    else if (body.detail !== undefined) message = JSON.stringify(body.detail);
  } catch {
    // not JSON: keep the status line
  }
  return new ApiError(message, response.status);
}

export function createClient(base = "", fetchImpl: typeof fetch = (...a) => fetch(...a)): ApiClient {
  async function call<T>(path: string, body?: unknown): Promise<T> {
    const response = await fetchImpl(`${base}${path}`, {
      method: body === undefined ? "GET" : "POST",
      headers: body === undefined ? undefined : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    if (!response.ok) throw await readError(response);
    return (await response.json()) as T;
  }

  return {
    catalog: () => call<Catalog>("/api/catalog"),
    validate: async (config) => (await call<{ errors: ValidationError[] }>("/api/config/validate", { config })).errors,
    parse: (yaml) => call<ParsedConfig>("/api/config/parse", { yaml }),
    dump: async (config, extras) => (await call<{ yaml: string }>("/api/config/dump", { config, extras })).yaml,
  };
}
