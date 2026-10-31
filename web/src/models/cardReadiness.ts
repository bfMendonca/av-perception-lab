// What a pipeline card shows of its models' readiness (the worst row decides).
import { formatBytes } from "../video/DropZone";
import { isUnknownSource, stateLabel, toneOf, where, worstRow, type Tone } from "./readiness";
import type { ModelView } from "./useModelStatus";

export interface CardReadiness {
  state: string;
  tone: Tone;
  /** Short state word on the card ("Not ready", "Failed", "Preparing", ...). */
  label: string;
  /** One line under the title (size to download, reason). */
  summary: string | null;
  /** Full reason (tooltip of the alert icon). */
  detail: string | null;
  /** Accessible name of the alert icon. */
  alertLabel: string;
  progress: { percent: number | null; text: string } | null;
}

function shortReason(message: string | null): string | null {
  if (!message) return null;
  const first = message.split(/;\s*/)[0];
  return first.length > 90 ? `${first.slice(0, 87)}…` : first;
}

export function cardReadiness(rows: ModelView[]): CardReadiness | undefined {
  const row = worstRow(rows);
  if (!row) return undefined;
  const tone = toneOf(row.state);
  const label = stateLabel(row.state);
  const others = rows.length > 1 ? ` (${rows.filter((r) => r.state !== "ready").length} of ${rows.length} models)` : "";
  let summary: string | null = null;
  let progress: CardReadiness["progress"] = null;
  if (tone === "preparing") {
    const p = row.progress;
    if (row.state === "queued") {
      progress = { percent: null, text: "Waiting in the prepare queue" };
      summary = "waiting in the queue";
    } else if (p && p.fraction !== null && p.total !== null) {
      const percent = Math.round(p.fraction * 100);
      const text = `${formatBytes(p.done)} of ${formatBytes(p.total)}`;
      progress = { percent, text };
      summary = text;
    } else {
      progress = { percent: null, text: p && p.done > 0 ? `${formatBytes(p.done)} downloaded` : "Preparing" };
      summary = p && p.done > 0 ? `${formatBytes(p.done)} downloaded` : "starting…";
    }
  } else if (tone !== "ready") {
    if (row.state === "missing" && !isUnknownSource(row) && !row.attempt && row.size_bytes) summary = `${formatBytes(row.size_bytes)} to download`;
    else summary = shortReason(row.message) ?? (row.state === "missing" ? "requirements missing" : null);
  }
  const detail = row.message ? `${where(row)} ${row.name}: ${row.message}` : null;
  return {
    state: row.state,
    tone,
    label,
    summary,
    detail,
    alertLabel: `${label}${others}${row.message ? `: ${row.message}` : ""}`,
    progress,
  };
}
