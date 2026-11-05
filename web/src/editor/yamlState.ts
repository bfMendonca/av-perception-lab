// Editor state: the parsed config (YAML layout) plus the top-level keys the
// server did not recognise (`extras`), a baseline for the diff, the current
// selection and the debounced server validation. Every rule about stages
// (single or multiple slots, optional) comes from the catalog.
import { createContext, createElement, useContext, useMemo, useSyncExternalStore, type ReactNode } from "react";
import type { ApiClient } from "../api/client";
import type { Catalog, ConfigDict, ParsedConfig, ValidationError } from "../api/types";
import { defaultsFromFields, moduleInfo, moduleTypes, sectionFields, stageByKey } from "./catalog";
import { deepEqual, diffTrees, getIn, isRecord, setIn, type DiffEntry, type Path } from "./paths";

export interface PipelineState {
  config: ConfigDict;
  extras: ConfigDict;
}

export type Selection = { kind: "slot"; stageKey: string; index: number | null } | { kind: "general"; stageKey?: string };

export type ValidationStatus = "idle" | "pending" | "valid" | "invalid" | "unavailable";

export interface Notice {
  tone: "info" | "warn" | "error";
  text: string;
}

export interface EditorState {
  catalog: Catalog | null;
  /** A pipeline was opened or started (the canvas shows lanes). */
  loaded: boolean;
  pipeline: PipelineState;
  /** Last opened or saved pipeline: the reference of `dirty` and `diff()`. */
  baseline: PipelineState;
  fileName: string | null;
  dirty: boolean;
  selection: Selection | null;
  errors: ValidationError[];
  validation: ValidationStatus;
  validationMessage: string | null;
  notice: Notice | null;
}

export interface Draft {
  pipeline: PipelineState;
  baseline: PipelineState;
  fileName: string | null;
}

export type Result = { ok: true } | { ok: false; error: string };

const EMPTY: PipelineState = { config: {}, extras: {} };

export interface StoreOptions {
  /** Delay of the server validation after the last change (ms). */
  debounceMs?: number;
}

export function createEditorStore(client: ApiClient, options: StoreOptions = {}) {
  const debounceMs = options.debounceMs ?? 400;
  let state: EditorState = {
    catalog: null,
    loaded: false,
    pipeline: EMPTY,
    baseline: EMPTY,
    fileName: null,
    dirty: false,
    selection: null,
    errors: [],
    validation: "idle",
    validationMessage: null,
    notice: null,
  };
  const listeners = new Set<() => void>();
  let timer: ReturnType<typeof setTimeout> | null = null;
  let validationSeq = 0;

  function set(patch: Partial<EditorState>) {
    state = { ...state, ...patch };
    listeners.forEach((l) => l());
  }

  function toConfig(pipeline: PipelineState = state.pipeline): ConfigDict {
    return { ...pipeline.config, ...pipeline.extras };
  }

  function commit(config: ConfigDict, extras: ConfigDict = state.pipeline.extras, patch: Partial<EditorState> = {}) {
    const pipeline = { config, extras };
    set({ pipeline, dirty: !deepEqual(toConfig(pipeline), toConfig(state.baseline)), notice: null, ...patch });
    scheduleValidation();
  }

  function reject(error: string): Result {
    set({ notice: { tone: "warn", text: error } });
    return { ok: false, error };
  }

  function scheduleValidation() {
    if (!state.loaded) return;
    if (timer !== null) clearTimeout(timer);
    validationSeq++; // a response to any request already in flight is now stale
    set({ validation: "pending" });
    timer = setTimeout(() => {
      timer = null;
      void validateNow();
    }, debounceMs);
  }

  async function validateNow(): Promise<void> {
    const seq = ++validationSeq;
    const config = toConfig();
    try {
      const errors = await client.validate(config);
      if (seq !== validationSeq) return;
      set({ errors, validation: errors.length ? "invalid" : "valid", validationMessage: null });
    } catch (exc) {
      if (seq !== validationSeq) return;
      set({ validation: "unavailable", validationMessage: exc instanceof Error ? exc.message : String(exc) });
    }
  }

  function specDefaults(kind: string, typeName: string): Record<string, unknown> {
    const info = moduleInfo(state.catalog, kind, typeName);
    return { type: typeName, ...(info ? defaultsFromFields(info.schema.fields) : {}) };
  }

  const store = {
    /** Debounce of the server checks (the model status follows it too). */
    debounceMs,
    getState: () => state,
    subscribe(listener: () => void) {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },

    setCatalog(catalog: Catalog) {
      set({ catalog });
    },

    loadParsed(parsed: ParsedConfig, fileName: string | null = null) {
      const pipeline = { config: parsed.config ?? {}, extras: parsed.extras ?? {} };
      set({
        loaded: true,
        pipeline,
        baseline: structuredClone(pipeline),
        fileName,
        dirty: false,
        selection: null,
        errors: [],
        notice: null,
      });
      scheduleValidation();
    },

    /**
     * Apply a preset: its pipeline replaces the current one, but the current
     * `source` and `output` stay (a preset is a recipe, not an input/output
     * choice; they are never created when absent). The editor is left clean.
     */
    loadPreset(parsed: ParsedConfig) {
      const config: ConfigDict = { ...(parsed.config ?? {}) };
      delete config.source;
      delete config.output;
      for (const key of ["source", "output"]) {
        if (key in state.pipeline.config) config[key] = state.pipeline.config[key];
      }
      store.loadParsed({ config, extras: parsed.extras ?? {} }, null);
    },

    /** Parse `yaml` on the server (unknown top-level keys become extras). */
    async load(yaml: string, fileName: string | null = null): Promise<void> {
      store.loadParsed(await client.parse(yaml), fileName);
    },

    /** Bring back a saved session (`snapshot()`), keeping its baseline. */
    restore(draft: Draft) {
      set({
        loaded: true,
        pipeline: draft.pipeline,
        baseline: draft.baseline,
        fileName: draft.fileName,
        dirty: !deepEqual(toConfig(draft.pipeline), toConfig(draft.baseline)),
        selection: null,
        errors: [],
        notice: null,
      });
      scheduleValidation();
    },

    snapshot: (): Draft | null =>
      state.loaded ? { pipeline: state.pipeline, baseline: state.baseline, fileName: state.fileName } : null,

    startEmpty() {
      store.loadParsed({ config: {}, extras: {} }, null);
    },

    toConfig: () => toConfig(),

    diff: (): DiffEntry[] => diffTrees(toConfig(state.baseline), toConfig()),

    /** The current pipeline becomes the baseline (after a save). */
    markSaved(fileName: string | null = state.fileName) {
      set({ baseline: structuredClone(state.pipeline), dirty: false, fileName });
    },

    select(selection: Selection | null) {
      set({ selection });
    },

    dismissNotice() {
      set({ notice: null });
    },

    notify(notice: Notice) {
      set({ notice });
    },

    /**
     * Multi-slot stage: append a spec (adapter stages: `{type, ...defaults}`).
     * Single-slot stage: fill the empty slot (an adapter spec, or the section's
     * defaults). Selects the new slot.
     */
    addModule(stageKey: string, typeName?: string): Result {
      const stage = stageByKey(state.catalog, stageKey);
      if (!stage) return reject(`Unknown stage "${stageKey}".`);
      let value: unknown;
      if (stage.kind !== null) {
        const type = typeName ?? moduleTypes(state.catalog, stage.kind)[0];
        if (!type) return reject(`No ${stage.kind} module is available.`);
        value = specDefaults(stage.kind, type);
      } else {
        value = defaultsFromFields(sectionFields(state.catalog, stageKey) ?? []);
      }
      const current = state.pipeline.config[stageKey];
      if (stage.multiple) {
        const list = Array.isArray(current) ? current : [];
        commit({ ...state.pipeline.config, [stageKey]: [...list, value] }, undefined, {
          selection: { kind: "slot", stageKey, index: list.length },
        });
        return { ok: true };
      }
      if (current !== undefined && current !== null) {
        return reject(`${stage.title} takes a single module; change its type in the inspector instead.`);
      }
      commit({ ...state.pipeline.config, [stageKey]: value }, undefined, {
        selection: { kind: "slot", stageKey, index: null },
      });
      return { ok: true };
    },

    /** Remove one spec of a multi-slot stage; single-slot stages are rejected. */
    removeModule(stageKey: string, index: number): Result {
      const stage = stageByKey(state.catalog, stageKey);
      if (!stage) return reject(`Unknown stage "${stageKey}".`);
      if (!stage.multiple) return reject(`${stage.title} has a single slot: it cannot be removed.`);
      const list = state.pipeline.config[stageKey];
      if (!Array.isArray(list) || index < 0 || index >= list.length) return reject("Nothing to remove.");
      const sel = state.selection;
      const keepSelection = sel?.kind === "slot" && sel.stageKey === stageKey ? null : sel;
      commit({ ...state.pipeline.config, [stageKey]: list.filter((_, i) => i !== index) }, undefined, {
        selection: keepSelection,
      });
      return { ok: true };
    },

    /** Turn an optional single-slot stage off (its key leaves the config). */
    disableStage(stageKey: string): Result {
      const stage = stageByKey(state.catalog, stageKey);
      if (!stage) return reject(`Unknown stage "${stageKey}".`);
      if (stage.multiple || !stage.optional) return reject(`${stage.title} cannot be turned off.`);
      const config = { ...state.pipeline.config };
      delete config[stageKey];
      commit(config, undefined, { selection: null });
      return { ok: true };
    },

    /** Switch the adapter of a slot: the new type's defaults, keeping the instance name. */
    changeType(stageKey: string, index: number | null, typeName: string): Result {
      const stage = stageByKey(state.catalog, stageKey);
      if (!stage || stage.kind === null) return reject("This stage has no module types.");
      const path: Path = index === null ? [stageKey] : [stageKey, index];
      const old = getIn(state.pipeline.config, path);
      const name = isRecord(old) ? old.name : undefined;
      const spec = { ...specDefaults(stage.kind, typeName), ...(name !== undefined ? { name } : {}) };
      commit(setIn(state.pipeline.config, path, spec) as ConfigDict);
      return { ok: true };
    },

    /** Set a value anywhere in the config; `undefined` removes the key. */
    setOption(path: Path, value: unknown) {
      if (path.length === 0) return;
      commit(setIn(state.pipeline.config, path, value) as ConfigDict);
    },

    /** Replace the config with `update(config)` in one change (nothing happens when it is equal). */
    updateConfig(update: (config: ConfigDict) => ConfigDict) {
      const next = update(state.pipeline.config);
      if (next !== state.pipeline.config && !deepEqual(next, state.pipeline.config)) commit(next);
    },

    /** Set (or remove, with `undefined`) a top-level extra key. */
    setExtra(key: string, value: unknown) {
      commit(state.pipeline.config, setIn(state.pipeline.extras, [key], value) as ConfigDict);
    },

    validateNow,
  };
  return store;
}

export type EditorStore = ReturnType<typeof createEditorStore>;

const EditorContext = createContext<EditorStore | null>(null);

export function EditorProvider({ store, children }: { store: EditorStore; children: ReactNode }) {
  return createElement(EditorContext.Provider, { value: store }, children);
}

export function useEditorStore(): EditorStore {
  const store = useContext(EditorContext);
  if (!store) throw new Error("useEditor() outside <EditorProvider>");
  return store;
}

/** The editor state plus its actions, re-rendering on every change. */
export function useEditor() {
  const store = useEditorStore();
  const state = useSyncExternalStore(store.subscribe, store.getState, store.getState);
  return useMemo(() => ({ ...store, ...state, state }), [store, state]);
}
