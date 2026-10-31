// Read-only helpers over the catalog. Stage and module names only ever come
// from the catalog itself.
import type { Catalog, Field, ModuleInfo, StageInfo } from "../api/types";
import { isRecord, type PathSeg } from "./paths";

/** Highest `catalog_version` this UI was written against; newer ones still render. */
export const SUPPORTED_CATALOG_VERSION = 1;

const byOrder = (a: StageInfo, b: StageInfo) => a.order - b.order;

export const laneStages = (c: Catalog) => c.stages.filter((s) => !s.general).sort(byOrder);
/** Lanes: consecutive lane stages sharing a non-empty `group` are merged. */
export interface LaneUnit {
  key: string;
  group: string | null;
  stages: StageInfo[];
}

export function laneUnits(c: Catalog): LaneUnit[] {
  const units: LaneUnit[] = [];
  for (const stage of laneStages(c)) {
    const group = typeof stage.group === "string" && stage.group ? stage.group : null;
    const last = units[units.length - 1];
    if (group !== null && last?.group === group) last.stages.push(stage);
    else units.push({ key: group !== null ? `group:${group}:${stage.yaml_key}` : stage.yaml_key, group, stages: [stage] });
  }
  return units;
}

export const generalStages = (c: Catalog) => c.stages.filter((s) => s.general).sort(byOrder);
export const stageByKey = (c: Catalog | null, key: string) => c?.stages.find((s) => s.yaml_key === key);

export function moduleInfo(c: Catalog | null, kind: string | null, type: unknown): ModuleInfo | undefined {
  if (!c || kind === null || typeof type !== "string") return undefined;
  return c.modules[kind]?.[type];
}

export const moduleTypes = (c: Catalog | null, kind: string | null): string[] =>
  c && kind !== null ? Object.keys(c.modules[kind] ?? {}) : [];

export const sectionFields = (c: Catalog | null, key: string): Field[] | undefined => c?.sections[key]?.schema.fields;

/** Values of the fields that have a default (required ones are left for the user). */
export function defaultsFromFields(fields: readonly Field[]): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const f of fields) if (!f.required) out[f.name] = structuredClone(f.default);
  return out;
}

export const fieldLabel = (f: Field) => (typeof f.hint?.label === "string" && f.hint.label ? f.hint.label : f.name);

/** Slots of a stage in the config: each spec with its path suffix (index or none). */
export interface Slot {
  index: number | null;
  value: unknown;
}

export function slotsOf(stage: StageInfo, value: unknown): Slot[] {
  if (stage.multiple) return Array.isArray(value) ? value.map((v, i) => ({ index: i, value: v })) : [];
  return value === undefined || value === null ? [] : [{ index: null, value }];
}

export const slotPath = (stage: StageInfo, index: number | null): PathSeg[] =>
  index === null ? [stage.yaml_key] : [stage.yaml_key, index];

/** Keys of an adapter spec that are not options: `type` and the optional instance `name`. */
export const SPEC_KEYS = ["type", "name"] as const;

export function formatValue(value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : String(Number(value.toPrecision(4)));
  if (typeof value === "string") return value === "" ? '""' : value;
  if (typeof value === "boolean") return value ? "on" : "off";
  if (Array.isArray(value)) return value.length === 0 ? "[]" : `[${value.length}]`;
  if (isRecord(value)) return `{${Object.keys(value).length}}`;
  return String(value);
}

/** A short card subtitle: up to `max` scalar options, labelled fields first. */
export function summarize(fields: readonly Field[] | undefined, value: unknown, max = 2): string {
  if (!isRecord(value)) return "";
  const scalar = (v: unknown) => v === null || ["string", "number", "boolean"].includes(typeof v);
  const known = new Set(fields?.map((f) => f.name));
  const ordered = [
    ...(fields ?? []).filter((f) => f.hint?.label && !f.hint.advanced),
    ...(fields ?? []).filter((f) => !f.hint?.label && !f.hint?.advanced),
  ].map((f) => f.name);
  const extra = Object.keys(value).filter((k) => !known.has(k) && !(SPEC_KEYS as readonly string[]).includes(k));
  const parts: string[] = [];
  for (const key of [...ordered, ...extra]) {
    if (parts.length >= max) break;
    if (!(key in value) || !scalar(value[key]) || value[key] === null) continue;
    parts.push(`${key} ${formatValue(value[key])}`);
  }
  return parts.join(" · ");
}

/**
 * Domain errors raised by an options dataclass (`__post_init__`) come back at
 * the spec's path (`detectors[0]: threshold must be ...`). When the message
 * starts with one of the schema's field names, attach it to that field.
 */
export function refineErrors<E extends { path: string; message: string }>(errors: E[], prefix: string, fields: readonly Field[] | undefined): E[] {
  if (!fields?.length) return errors;
  const names = new Set(fields.map((f) => f.name));
  return errors.map((e) => {
    if (e.path !== prefix) return e;
    const first = /^([A-Za-z_]\w*)\b/.exec(e.message)?.[1];
    return first && names.has(first) ? { ...e, path: prefix ? `${prefix}.${first}` : first } : e;
  });
}
