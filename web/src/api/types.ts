// Shapes of the av-perception-lab web API (src/av_perception_lab/web/schema.py).
// Everything here is data: the UI never names a stage, a model or a field.

/** Field types the server emits today; any other string is rendered generically. */
export type KnownFieldType =
  | "bool"
  | "int"
  | "float"
  | "str"
  | "path"
  | "enum"
  | "list"
  | "map"
  | "object"
  | "any";

export type FieldType = KnownFieldType | (string & {});

/** UI hints from dataclass field metadata (`metadata={"ui": {...}}`); all optional. */
export interface FieldHint {
  label?: string;
  unit?: string;
  min?: number;
  max?: number;
  step?: number;
  advanced?: boolean;
  help?: string;
  [key: string]: unknown;
}

/** Schema of one annotation (a field, a list item, a map value or a variant). */
export interface TypeSchema {
  type: FieldType;
  nullable?: boolean;
  choices?: unknown[];
  items?: TypeSchema;
  fields?: Field[];
  variants?: TypeSchema[];
  length?: number;
  [key: string]: unknown;
}

export interface Field extends TypeSchema {
  name: string;
  default: unknown;
  required: boolean;
  hint?: FieldHint;
}

export interface OptionsSchema {
  fields: Field[];
}

export interface StageInfo {
  yaml_key: string;
  title: string;
  /** Adapter family of the stage's specs, or null for a plain config section. */
  kind: string | null;
  multiple: boolean;
  optional: boolean;
  order: number;
  /** Shown under "General settings" instead of a lane. */
  general: boolean;
  has_section: boolean;
  /** Consecutive stages with the same group share one lane titled by it. */
  group?: string | null;
  [key: string]: unknown;
}

export interface ModuleInfo {
  schema: OptionsSchema;
  doc?: string;
  [key: string]: unknown;
}

export interface Catalog {
  catalog_version: number;
  stages: StageInfo[];
  /** kind -> type_name -> module */
  modules: Record<string, Record<string, ModuleInfo>>;
  /** yaml_key -> section schema */
  sections: Record<string, { schema: OptionsSchema }>;
  [key: string]: unknown;
}

export interface ValidationError {
  /** Dotted path, e.g. `detectors[0].threshold`; empty when global. */
  path: string;
  message: string;
}

export type ConfigDict = Record<string, unknown>;

export interface ParsedConfig {
  config: ConfigDict;
  extras: ConfigDict;
}
