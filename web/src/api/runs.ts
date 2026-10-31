// Client of the results routes (src/det_depth_distance/web/app.py, web/compare.py):
// summaries, comparison, distance series, discovered artifacts, reopen, rerun.
// Injected so tests can fake it. Metric names, labels and the "better"
// direction come from the server (`metrics`), never from this file.
import type { ParsedConfig } from "./types";
import { ApiError } from "./client";

/** One summary value as the server describes it. */
export interface MetricInfo {
  key: string;
  label: string;
  unit: string | null;
  /** Which direction is an improvement; null = neither. */
  better: "lower" | "higher" | null;
  help?: string;
}

export type Summary = Record<string, number | null | string[] | undefined> & { notes?: string[] };

export interface SummaryResponse {
  summary: Summary;
  metrics: MetricInfo[];
}

export interface Delta {
  abs: number;
  rel: number | null;
}

export interface DiffEntry {
  path: string;
  a: unknown;
  b: unknown;
  change: "changed" | "added" | "removed" | (string & {});
  /** Set by the server for every web job (run folder, tag): always differs. */
  server_set?: boolean;
}

export interface Comparison {
  a: string;
  b: string;
  metrics: MetricInfo[];
  summary_a: Summary;
  summary_b: Summary;
  deltas: Record<string, Delta>;
  /** null when one of the runs recorded no configuration. */
  config_diff: DiffEntry[] | null;
  config_origin: { a: string | null; b: string | null };
}

export interface TrackInfo {
  id: number;
  rows: number;
  t0: number;
  t1: number;
  class_name: string | null;
}

export interface Series {
  track: number | null;
  found: boolean;
  tracks: TrackInfo[];
  /** Frame.t in seconds. */
  t: number[];
  raw: (number | null)[];
  smoothed: (number | null)[] | null;
  /** Time of each point in the annotated video (what a click seeks to). */
  video_t: (number | null)[];
  points: number;
  downsampled: boolean;
  note?: string | null;
}

export interface Artifact {
  name: string;
  size: number;
  media_type: string;
  modified: string;
  /** A file the UI knows (`video`, `detections`, `metrics`, ...); null otherwise. */
  role: string | null;
  url: string;
}

export interface Reopened extends ParsedConfig {
  origin: string;
  file_name: string;
}

export interface RunsApi {
  summary(runId: string): Promise<SummaryResponse>;
  compare(a: string, b: string): Promise<Comparison>;
  series(runId: string, track?: number | null): Promise<Series>;
  artifacts(runId: string): Promise<Artifact[]>;
  /** JSON of an artifact URL (as listed by `artifacts`). */
  fetchJson(url: string): Promise<unknown>;
  reopen(runId: string): Promise<Reopened>;
  /** Queue the run again; resolves the job id. */
  rerun(runId: string): Promise<string>;
}

type NotReadyRow = { stage: string; index: number | null; name: string; state: string };

/** "2 models are not ready: <stage>[0] x (missing), <stage> y (failed). Prepare them in the editor first." */
export function notReadyText(rows: NotReadyRow[]): string {
  const list = rows.map((r) => `${r.index === null ? r.stage : `${r.stage}[${r.index}]`} ${r.name} (${r.state})`).join(", ");
  return `${rows.length} model${rows.length === 1 ? " is" : "s are"} not ready: ${list}. Prepare ${rows.length === 1 ? "it" : "them"} in the editor first.`;
}

async function errorOf(response: Response): Promise<ApiError> {
  let message = `${response.status} ${response.statusText}`.trim();
  try {
    const body = (await response.json()) as { detail?: unknown; errors?: { path: string; message: string }[]; models_not_ready?: NotReadyRow[] };
    if (typeof body.detail === "string") message = body.detail;
    else if (Array.isArray(body.models_not_ready)) message = notReadyText(body.models_not_ready);
    else if (body.errors?.length) message = body.errors.map((e) => (e.path ? `${e.path}: ${e.message}` : e.message)).join("; ");
  } catch {
    // keep the status line
  }
  return new ApiError(message, response.status);
}

export function createRunsApi(base = "", fetchImpl: typeof fetch = (...a) => fetch(...a)): RunsApi {
  async function call<T>(path: string, init?: RequestInit): Promise<T> {
    const response = await fetchImpl(`${base}${path}`, init);
    if (!response.ok) throw await errorOf(response);
    return (await response.json()) as T;
  }
  const enc = encodeURIComponent;
  const post: RequestInit = { method: "POST" };
  return {
    summary: (id) => call<SummaryResponse>(`/api/runs/${enc(id)}/summary`),
    compare: (a, b) => call<Comparison>(`/api/compare?a=${enc(a)}&b=${enc(b)}`),
    series: (id, track) => call<Series>(`/api/runs/${enc(id)}/series${track === null || track === undefined ? "" : `?track=${track}`}`),
    artifacts: (id) => call<Artifact[]>(`/api/runs/${enc(id)}/artifacts`),
    fetchJson: (url) => call<unknown>(url),
    reopen: (id) => call<Reopened>(`/api/runs/${enc(id)}/reopen`, post),
    rerun: async (id) => (await call<{ job_id: string }>(`/api/runs/${enc(id)}/rerun`, post)).job_id,
  };
}
