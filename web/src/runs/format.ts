// Number, size, date and delta formatting of the runs view. Generic: the
// metric (label, unit, better direction) is described by the server.
import type { Delta, MetricInfo } from "../api/runs";

const MINUS = "−";

function digits(value: number): number {
  const a = Math.abs(value);
  if (a === 0 || Number.isInteger(value)) return 0;
  if (a >= 100) return 0;
  if (a >= 10) return 1;
  if (a >= 1) return 2;
  return 3;
}

export function formatNumber(value: number, fractionDigits = digits(value)): string {
  const text = Math.abs(value).toLocaleString("en-US", { maximumFractionDigits: fractionDigits, minimumFractionDigits: Math.min(fractionDigits, Number.isInteger(value) ? 0 : 1) });
  return value < 0 ? `${MINUS}${text}` : text;
}

export function formatValue(value: unknown, unit: string | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "n/a";
  return unit ? `${formatNumber(value)} ${unit}` : formatNumber(value);
}

export type DeltaTone = "good" | "bad" | "neutral";

export interface DeltaView {
  text: string;
  tone: DeltaTone;
  /** Accessible wording, e.g. "17.6 % lower than the comparison run (better)". */
  label: string;
}

/** Signed delta of B against A, and whether it is an improvement. */
export function deltaView(metric: MetricInfo, delta: Delta | undefined): DeltaView | null {
  if (!delta || !Number.isFinite(delta.abs)) return null;
  const sign = delta.abs > 0 ? "+" : delta.abs < 0 ? MINUS : "±";
  const rel = delta.rel !== null && Number.isFinite(delta.rel) ? delta.rel * 100 : null;
  const text = rel !== null ? `${sign}${formatNumber(Math.abs(rel), Math.abs(rel) < 10 ? 1 : 0)} %` : `${sign}${formatNumber(Math.abs(delta.abs))}${metric.unit ? ` ${metric.unit}` : ""}`;
  let tone: DeltaTone = "neutral";
  if (metric.better && delta.abs !== 0) {
    const lower = delta.abs < 0;
    tone = (metric.better === "lower") === lower ? "good" : "bad";
  }
  const direction = delta.abs > 0 ? "higher" : delta.abs < 0 ? "lower" : "unchanged";
  const label = `${text.replace(MINUS, "")} ${direction} than the comparison run${tone === "good" ? " (better)" : tone === "bad" ? " (worse)" : ""}`;
  return { text, tone, label };
}

export function formatBytes(size: number): string {
  if (size < 1024) return `${size} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let v = size / 1024;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i += 1;
  }
  return `${v.toFixed(v >= 100 ? 0 : 1)} ${units[i]}`;
}

export function formatWhen(iso: string): string {
  const d = new Date(iso);
  if (!iso || Number.isNaN(d.getTime())) return iso || "unknown date";
  const today = new Date();
  const sameDay = d.toDateString() === today.toDateString();
  return sameDay ? `today ${d.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" })}` : d.toLocaleString("en-GB", { day: "numeric", month: "short", year: d.getFullYear() === today.getFullYear() ? undefined : "numeric", hour: "2-digit", minute: "2-digit" });
}

export function formatSeconds(s: number): string {
  if (!Number.isFinite(s)) return "–";
  const m = Math.floor(s / 60);
  const rest = s - m * 60;
  return m ? `${m}:${rest.toFixed(1).padStart(4, "0")}` : `${rest.toFixed(1)} s`;
}
