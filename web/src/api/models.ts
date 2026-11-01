// Client of the model readiness routes (src/det_depth_distance/web/models.py):
// status of a config's models, the Prepare queue and its SSE events. Rows are
// data: the UI never names a model, a stage or a source.
import { ApiError } from "./client";
import type { ConfigDict, ValidationError } from "./types";

/** States the server emits today; any other string renders with the generic not-ready style. */
export type KnownModelState = "ready" | "missing" | "conflict" | "failed" | "preparing";
export type ModelState = KnownModelState | (string & {});

/** Where a model sits in the config: the path `<stage>[1]` is `{stage, index: 1}`; a single slot has index null. */
export interface ModelKey {
  stage: string;
  index: number | null;
}

/** One row of `POST /api/models/status`. Optional extra fields are shown when present. */
export interface ModelStatus extends ModelKey {
  type: string;
  name: string;
  state: ModelState;
  /** `assets` / `code` / `packages` -> ready | missing | partial | invalid. */
  layers: Record<string, string>;
  /** What is not ready: asset keys, `code:<name>`, `package:<name>`. */
  missing: string[];
  /** Bytes still to download (declared sizes), null when unknown. */
  size_bytes: number | null;
  message: string | null;
  /** Declared licences ("; "-joined), null when none. */
  license?: string | null;
  /** Where the files come from, one line per kind ("; "-joined). */
  source?: string | null;
  /** Folders/files relative to the weights root ("; "-joined). */
  target?: string | null;
  /** Why the web cannot prepare it: "unknown_source", "packages_only" (or another server word), null otherwise. */
  reason?: string | null;
  /** A Prepare from the web can make it ready. Absent from older servers (then the UI infers it). */
  preparable?: boolean;
  [key: string]: unknown;
}

export interface StatusReport {
  models: ModelStatus[];
  total_size_bytes: number;
}

/** One model inside a Prepare job. */
export interface PrepareModel extends ModelStatus {
  /** queued | preparing | ready | missing | failed | cancelled | ... */
  item: string | null;
  done: number;
  total: number | null;
}

export type PrepareStatus = "queued" | "running" | "done" | "failed" | "cancelled" | (string & {});

export interface PrepareJob {
  id: string;
  status: PrepareStatus;
  position: number | null;
  error: string | null;
  created: string;
  cancel_requested: boolean;
  models: PrepareModel[];
  /** Models of a `models: "all"` request left out (unknown source). */
  skipped: (ModelKey & { type: string; name: string; message: string })[];
}

/** One server-sent event of a Prepare job (`queued`, `running`, `model`, `progress`, `cancelling`, `end`, ...). */
export interface PrepareEvent {
  event: string;
  [key: string]: unknown;
}

/** The config is invalid for the status/prepare routes (422 with `errors`). */
export class StatusRejected extends Error {
  constructor(readonly errors: ValidationError[]) {
    super(errors.length ? `${errors[0].path ? `${errors[0].path}: ` : ""}${errors[0].message}` : "The configuration is invalid.");
    this.name = "StatusRejected";
  }
}

export interface ModelsApi {
  status(config: ConfigDict): Promise<StatusReport>;
  prepare(models: ModelKey[] | "all", config: ConfigDict): Promise<PrepareJob>;
  getPrepare(id: string): Promise<PrepareJob>;
  /** Unfinished and recent Prepare jobs of the server, newest first. */
  listPrepares(): Promise<PrepareJob[]>;
  cancelPrepare(id: string): Promise<PrepareJob>;
  /** Subscribe to a job's events (replayed from the start); returns the unsubscribe. */
  prepareEvents(id: string, onEvent: (event: PrepareEvent) => void, onError?: () => void): () => void;
}

async function failure(response: Response): Promise<Error> {
  let message = `${response.status} ${response.statusText}`.trim();
  try {
    const body = (await response.json()) as { detail?: unknown; errors?: ValidationError[] };
    if (response.status === 422 && Array.isArray(body.errors)) return new StatusRejected(body.errors);
    if (typeof body.detail === "string") message = body.detail;
  } catch {
    // not JSON: keep the status line
  }
  return new ApiError(message, response.status);
}

export function createModelsApi(base = "", fetchImpl: typeof fetch = (...a) => fetch(...a)): ModelsApi {
  async function call<T>(path: string, init?: RequestInit): Promise<T> {
    const response = await fetchImpl(`${base}${path}`, init);
    if (!response.ok) throw await failure(response);
    return (await response.json()) as T;
  }
  const post = (body: unknown): RequestInit => ({ method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const enc = encodeURIComponent;

  return {
    status: (config) => call<StatusReport>("/api/models/status", post({ config })),
    prepare: (models, config) => call<PrepareJob>("/api/models/prepare", post({ models, config })),
    getPrepare: (id) => call<PrepareJob>(`/api/models/prepare/${enc(id)}`),
    listPrepares: () => call<PrepareJob[]>("/api/models/prepare"),
    cancelPrepare: (id) => call<PrepareJob>(`/api/models/prepare/${enc(id)}`, { method: "DELETE" }),
    prepareEvents: (id, onEvent, onError) => {
      const source = new EventSource(`${base}/api/models/prepare/${enc(id)}/events`);
      let ended = false;
      source.onmessage = (e: MessageEvent<string>) => {
        try {
          const event = JSON.parse(e.data) as PrepareEvent;
          if (event.event === "end") {
            ended = true;
            source.close(); // otherwise EventSource reconnects and replays
          }
          onEvent(event);
        } catch {
          // a malformed line is ignored
        }
      };
      source.onerror = () => {
        if (!ended && source.readyState === EventSource.CLOSED) onError?.();
      };
      return () => source.close();
    },
  };
}
