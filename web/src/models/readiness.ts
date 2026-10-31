// Pure rules of the readiness UI: which visual tone a state gets, which rows
// belong to a card, what a Prepare can fix. Everything is read from the rows
// the server sends; no model, stage or source name appears here.
import type { ModelKey, ModelState, ModelStatus } from "../api/models";
import { formatBytes } from "../video/DropZone";

/** Visual tone: ready (solid), preparing (bar), failed (red dashed), notready (yellow dashed: missing, conflict, any other state). */
export type Tone = "ready" | "preparing" | "failed" | "notready";

/** Message prefix the server uses for a source the web may not download. */
export const UNKNOWN_SOURCE_PREFIX = "unknown source";
/** `missing` entries of the packages layer. */
export const PACKAGE_PREFIX = "package:";
/** Label the server uses instead of the absolute weights root. */
export const ROOT_LABEL = "<weights root>";

export function toneOf(state: ModelState): Tone {
  if (state === "ready") return "ready";
  if (state === "preparing" || state === "queued") return "preparing";
  if (state === "failed") return "failed";
  return "notready";
}

export const keyOf = (k: ModelKey): string => `${k.stage}|${k.index ?? ""}`;

/** `<stage>[<index>]` or `<stage>`: how the config path names the model. */
export const where = (k: ModelKey): string => (k.index === null ? k.stage : `${k.stage}[${k.index}]`);

/** Human state label ("Not ready", "Failed", "Preparing", or the server's word for anything else). */
export function stateLabel(state: ModelState): string {
  switch (state) {
    case "ready":
      return "Ready";
    case "missing":
      return "Not ready";
    case "preparing":
      return "Preparing";
    case "queued":
      return "Queued";
    case "failed":
      return "Failed";
    case "conflict":
      return "Conflict";
    default:
      return state ? state.charAt(0).toUpperCase() + state.slice(1) : "Unknown";
  }
}

export const isUnknownSource = (row: Pick<ModelStatus, "message">): boolean =>
  typeof row.message === "string" && row.message.startsWith(UNKNOWN_SOURCE_PREFIX);

/** Only Python packages are missing: Prepare (weights and code) cannot fix it. */
export const onlyPackages = (row: Pick<ModelStatus, "missing">): boolean =>
  row.missing.length > 0 && row.missing.every((m) => m.startsWith(PACKAGE_PREFIX));

/** A Prepare from the web can make this row ready (or try again). */
export function preparable(row: Pick<ModelStatus, "state" | "message" | "missing">): boolean {
  if (row.state === "ready" || row.state === "preparing" || row.state === "queued" || row.state === "conflict") return false;
  return !isUnknownSource(row) && !onlyPackages(row);
}

/**
 * Rows shown on the card (or in the inspector) of a slot: the rows of that
 * stage and index, plus — for a single-slot stage — the rows of its
 * sub-paths (`<stage>.<part>` under `<stage>`).
 */
export function rowsForSlot<T extends ModelKey>(rows: T[], stageKey: string, index: number | null): T[] {
  return rows.filter((r) => (r.stage === stageKey && r.index === index) || (index === null && r.stage.startsWith(`${stageKey}.`)));
}

const RANK: Tone[] = ["ready", "notready", "preparing", "failed"];

/** The row that decides a card's look when several models share it. */
export function worstRow<T extends { state: ModelState }>(rows: T[]): T | undefined {
  return rows.reduce<T | undefined>((worst, r) => (!worst || RANK.indexOf(toneOf(r.state)) > RANK.indexOf(toneOf(worst.state)) ? r : worst), undefined);
}

/** Paths the message names under the weights root (where files go). */
export function locations(message: string | null | undefined): string[] {
  if (!message) return [];
  const found = message.match(/<weights root>[^\s;,)'"`]*/g) ?? [];
  return [...new Set(found.map((p) => p.replace(/[.:]+$/, "")))];
}

/** "690 MB", "unknown size" or "at least 1.2 GB" for a set of rows. */
export function sizeOf(rows: Pick<ModelStatus, "size_bytes">[]): { bytes: number; known: boolean; text: string } {
  const bytes = rows.reduce((s, r) => s + (r.size_bytes ?? 0), 0);
  const known = rows.every((r) => r.size_bytes !== null);
  if (!bytes) return { bytes, known, text: known ? "nothing to download" : "unknown size" };
  return { bytes, known, text: known ? formatBytes(bytes) : `at least ${formatBytes(bytes)}` };
}

/** The CLI command that prepares any model (sources the web does not vouch for too). */
export function prepareCommand(fileName: string | null): string {
  return `det-depth-distance --config ${fileName ?? "<config.yaml>"} --prepare`;
}

export const plural = (n: number, one: string, many: string = `${one}s`) => `${n} ${n === 1 ? one : many}`;
