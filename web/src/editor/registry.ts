// Front-end extension registry: optional custom views; the generic renderers
// are the fallbacks. Registering returns a function that undoes it.
import { createContext, type ReactElement } from "react";
import type { Catalog, Field, StageInfo, ValidationError } from "../api/types";
import type { Path } from "./paths";
import type { Selection } from "./yamlState";

export interface FieldControlProps {
  field: Field;
  value: unknown;
  onChange: (value: unknown) => void;
  path: Path;
  /** Server errors at this field's path (or under it, for leaf-like controls). */
  errors: ValidationError[];
  /** DOM id of the control, for the label. */
  id: string;
}

export type FieldControl = (props: FieldControlProps) => ReactElement | null;

export interface StageViewProps {
  stage: StageInfo;
  value: unknown;
  catalog: Catalog;
  errors: ValidationError[];
  selection: Selection | null;
  select: (selection: Selection) => void;
}

export type StageView = (props: StageViewProps) => ReactElement | null;

/**
 * Data of the app that custom stage views may read (the current input, for
 * one); the editor passes it along without looking inside.
 */
export type StageContextValue = Readonly<Record<string, unknown>>;
export const StageContext = createContext<StageContextValue>({});

export interface StageViewOptions {
  /** Leave the stage's lane out of the canvas (its value and the app's StageContext decide). */
  hidden?: (value: unknown, context: StageContextValue) => boolean;
}

const stageViews = new Map<string, StageView>();
const stageOptions = new Map<StageView, StageViewOptions>();
const fieldControlEntries: { match: (f: Field) => boolean; control: FieldControl }[] = [];

/** Replace the generic lane of the stage `yamlKey`. */
export function registerStageView(yamlKey: string, view: StageView, options: StageViewOptions = {}): () => void {
  stageViews.set(yamlKey, view);
  stageOptions.set(view, options);
  return () => {
    if (stageViews.get(yamlKey) === view) stageViews.delete(yamlKey);
  };
}

export const stageViewFor = (yamlKey: string): StageView | undefined => stageViews.get(yamlKey);

/** A registered view asks to leave the stage `yamlKey` out of the canvas. */
export function stageHidden(yamlKey: string, value: unknown, context: StageContextValue): boolean {
  const view = stageViews.get(yamlKey);
  return Boolean(view && stageOptions.get(view)?.hidden?.(value, context));
}

/** Use `control` for every field `match` accepts; the latest registration wins. */
export function registerFieldControl(match: (f: Field) => boolean, control: FieldControl): () => void {
  const entry = { match, control };
  fieldControlEntries.unshift(entry);
  return () => {
    const i = fieldControlEntries.indexOf(entry);
    if (i >= 0) fieldControlEntries.splice(i, 1);
  };
}

export function customFieldControl(field: Field): FieldControl | undefined {
  return fieldControlEntries.find((e) => e.match(field))?.control;
}
