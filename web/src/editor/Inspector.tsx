// Right-hand panel: the configuration of the selected card, or the general
// settings. Fields come from the catalog schema; keys the schema does not
// describe (and modules this server does not know) are edited as raw YAML.
import { useState, type ReactNode } from "react";
import type { Catalog, Field, StageInfo, ValidationError } from "../api/types";
import { Icon } from "../ui/icons";
import { generalStages, moduleInfo, moduleTypes, refineErrors, SPEC_KEYS, sectionFields, slotPath, stageByKey } from "./catalog";
import { FieldList, fieldControls } from "./fieldControls";
import { getIn, isRecord, isUnder, toDotted, type Path } from "./paths";
import { useEditor, type Selection } from "./yamlState";

const rawField = (name: string, value: unknown): Field => ({ name, type: "any", nullable: true, default: value, required: false, hint: {} });

export function Inspector() {
  const { catalog, selection, loaded } = useEditor();
  if (!catalog || !loaded) return <InspectorEmpty title="No pipeline yet" text="Open a YAML file or start an empty pipeline to configure its modules." />;
  if (!selection) return <InspectorEmpty title="Nothing selected" text="Click a module in the lanes to configure it, or open General settings." />;
  if (selection.kind === "general") return <GeneralSettings catalog={catalog} active={selection.stageKey} />;
  const stage = stageByKey(catalog, selection.stageKey);
  if (!stage) return <InspectorEmpty title="Unknown stage" text={`"${selection.stageKey}" is not in this server's catalog.`} />;
  return stage.kind !== null ? <SlotInspector catalog={catalog} stage={stage} index={selection.index} /> : <SectionInspector catalog={catalog} stage={stage} />;
}

function InspectorEmpty({ title, text }: { title: string; text: string }) {
  return (
    <div className="insp-empty">
      <div className="insp-empty__art" aria-hidden="true">
        <span />
        <span />
        <span />
      </div>
      <h2>{title}</h2>
      <p>{text}</p>
    </div>
  );
}

function Problems({ errors, prefix }: { errors: ValidationError[]; prefix: string }) {
  if (!errors.length) return null;
  return (
    <div className="problems" role="group" aria-label="Problems">
      <div className="problems__head">
        <Icon name="alert" size={14} />
        {errors.length} problem{errors.length > 1 ? "s" : ""}
      </div>
      <ul>
        {errors.map((e) => (
          <li key={`${e.path}:${e.message}`}>
            <code>{e.path.slice(prefix.length).replace(/^\./, "") || e.path || "config"}</code> {e.message}
          </li>
        ))}
      </ul>
    </div>
  );
}

function InspectorHead({ title, sub, doc, actions }: { title: string; sub: ReactNode; doc?: string; actions?: ReactNode }) {
  return (
    <header className="insp__head">
      <div className="insp__titles">
        <h2 className="insp__title">{title}</h2>
        <div className="insp__sub">{sub}</div>
      </div>
      {actions ? <div className="insp__actions">{actions}</div> : null}
      {doc ? <p className="insp__doc">{doc}</p> : null}
    </header>
  );
}

/** Spec keys the module schema does not describe, each as raw YAML with a remove button. */
function UnknownKeys({ value, known, base, errors, onChange }: { value: Record<string, unknown>; known: Set<string>; base: Path; errors: ValidationError[]; onChange: (key: string, v: unknown) => void }) {
  const keys = Object.keys(value).filter((k) => !known.has(k));
  if (!keys.length) return null;
  return (
    <div className="unknown-keys">
      <h3 className="insp__section">Keys not in the schema</h3>
      {keys.map((k) => (
        <div className="unknown-keys__row" key={k}>
          {fieldControls(rawField(k, value[k]), { value: value[k], onChange: (v) => onChange(k, v), path: [...base, k], errors })}
          <button type="button" className="icon-btn" aria-label={`Remove key ${k}`} title="Remove key" onClick={() => onChange(k, undefined)}>
            <Icon name="trash" />
          </button>
        </div>
      ))}
    </div>
  );
}

function SlotInspector({ catalog, stage, index }: { catalog: Catalog; stage: StageInfo; index: number | null }) {
  const editor = useEditor();
  const path = slotPath(stage, index);
  const dotted = toDotted(path);
  const spec = getIn(editor.pipeline.config, path);
  const kind = stage.kind as string;
  const type = isRecord(spec) && typeof spec.type === "string" ? spec.type : undefined;
  const info = moduleInfo(catalog, kind, type);
  const errors = refineErrors(
    editor.errors.filter((e) => isUnder(e.path, dotted)),
    dotted,
    info?.schema.fields,
  );
  const types = moduleTypes(catalog, kind);
  const set = (key: string, v: unknown) => editor.setOption([...path, key], v);
  const [nameDraft, setNameDraft] = useState<string | null>(null);

  if (spec === undefined) return <InspectorEmpty title="Removed" text="This module is no longer in the pipeline." />;

  const actions = stage.multiple ? (
    <button type="button" className="btn btn--danger btn--sm" onClick={() => editor.removeModule(stage.yaml_key, index as number)}>
      <Icon name="trash" size={14} /> Remove
    </button>
  ) : stage.optional ? (
    <button type="button" className="btn btn--ghost btn--sm" onClick={() => editor.disableStage(stage.yaml_key)}>
      Turn off
    </button>
  ) : null;

  const sub = (
    <>
      {stage.title} · <code>{dotted}</code>
    </>
  );

  if (!isRecord(spec) || !info) {
    return (
      <div className="insp">
        <InspectorHead title={type ?? stage.title} sub={sub} actions={actions} />
        <Problems errors={errors} prefix={dotted} />
        <p className="insp__note">
          <Icon name="info" size={14} /> {type ? `"${type}" is not a ${kind} module this server knows.` : "This entry is not a module spec."} Edit it as YAML.
        </p>
        {fieldControls(rawField(dotted, spec), { value: spec, onChange: (v) => editor.setOption(path, v), path, errors })}
        {types.length ? <TypePicker types={types} value={type} onPick={(t) => editor.changeType(stage.yaml_key, index, t)} /> : null}
      </div>
    );
  }

  const fields = info.schema.fields;
  const known = new Set<string>([...SPEC_KEYS, ...fields.map((f) => f.name)]);
  const name = typeof spec.name === "string" ? spec.name : "";
  const nameErrors = errors.filter((e) => e.path === `${dotted}.name`);

  return (
    <div className="insp">
      <InspectorHead title={type as string} sub={sub} doc={info.doc} actions={actions} />
      <Problems errors={errors} prefix={dotted} />
      <div className="insp__meta">
        <TypePicker types={types} value={type} onPick={(t) => t !== type && editor.changeType(stage.yaml_key, index, t)} />
        <div className={`field${nameErrors.length ? " field--invalid" : ""}`} data-testid={`field:${dotted}.name`}>
          <div className="field__head">
            <label className="field__label" htmlFor={`${dotted}-name`}>
              Instance name
            </label>
            <code className="field__key">name</code>
          </div>
          <input
            id={`${dotted}-name`}
            className="input input--mono"
            placeholder="optional"
            spellCheck={false}
            value={nameDraft ?? name}
            aria-invalid={nameErrors.length ? true : undefined}
            onChange={(e) => setNameDraft(e.target.value)}
            onBlur={() => {
              if (nameDraft !== null && nameDraft !== name) set("name", nameDraft === "" ? undefined : nameDraft);
              setNameDraft(null);
            }}
          />
          {nameErrors.map((e) => (
            <p className="field__local-error" key={e.message}>
              {e.message}
            </p>
          ))}
        </div>
      </div>
      <h3 className="insp__section">Options</h3>
      {fields.length ? <FieldList fields={fields} value={spec} basePath={path} errors={errors} onChange={set} /> : <p className="insp__note">This module has no options.</p>}
      <UnknownKeys value={spec} known={known} base={path} errors={errors} onChange={set} />
    </div>
  );
}

function TypePicker({ types, value, onPick }: { types: string[]; value: string | undefined; onPick: (t: string) => void }) {
  return (
    <div className="field">
      <div className="field__head">
        <label className="field__label" htmlFor="module-type" title="Switching resets the options to the new module's defaults">
          Module
        </label>
        <code className="field__key">type</code>
      </div>
      <div className="select">
        <select id="module-type" className="input select__el" value={value && types.includes(value) ? value : ""} onChange={(e) => onPick(e.target.value)}>
          {value && types.includes(value) ? null : <option value="">choose…</option>}
          {types.map((t) => (
            <option key={t} value={t}>
              {t}
            </option>
          ))}
        </select>
        <Icon name="chevron" className="select__icon" />
      </div>
    </div>
  );
}

/** The fields of one config section (a lane's section card or a general tab). */
function SectionBody({ catalog, stage }: { catalog: Catalog; stage: StageInfo }) {
  const editor = useEditor();
  const key = stage.yaml_key;
  const value = editor.pipeline.config[key];
  const fields = sectionFields(catalog, key);
  const errors = refineErrors(
    editor.errors.filter((e) => isUnder(e.path, key)),
    key,
    fields,
  );
  if (value === null && stage.optional) {
    return (
      <div className="unset unset--block">
        <span>{stage.title} is off.</span>
        <button type="button" className="btn btn--sm" onClick={() => {
            editor.setOption([key], undefined);
            editor.addModule(key);
          }}
        >
          <Icon name="plus" size={14} /> Enable
        </button>
      </div>
    );
  }
  if (!fields || (value !== undefined && !isRecord(value))) {
    return fieldControls(rawField(key, value), { value, onChange: (v) => editor.setOption([key], v), path: [key], errors });
  }
  const obj = isRecord(value) ? value : {};
  const known = new Set(fields.map((f) => f.name));
  return (
    <>
      <Problems errors={errors} prefix={key} />
      <FieldList fields={fields} value={obj} basePath={[key]} errors={errors} onChange={(name, v) => editor.setOption([key, name], v)} />
      <UnknownKeys value={obj} known={known} base={[key]} errors={errors} onChange={(k, v) => editor.setOption([key, k], v)} />
    </>
  );
}

function SectionInspector({ catalog, stage }: { catalog: Catalog; stage: StageInfo }) {
  const editor = useEditor();
  const actions = stage.optional ? (
    <button type="button" className="btn btn--ghost btn--sm" onClick={() => editor.disableStage(stage.yaml_key)}>
      Turn off
    </button>
  ) : null;
  return (
    <div className="insp">
      <InspectorHead title={stage.title} sub={<code>{stage.yaml_key}</code>} actions={actions} />
      <SectionBody catalog={catalog} stage={stage} />
    </div>
  );
}

const OTHER = "__other__";

/** General settings: one tab per `general` stage, plus top-level keys no stage owns and the extras. */
function GeneralSettings({ catalog, active }: { catalog: Catalog; active?: string }) {
  const editor = useEditor();
  const tabs = generalStages(catalog);
  const owned = new Set(catalog.stages.map((s) => s.yaml_key));
  const otherKeys = Object.keys(editor.pipeline.config).filter((k) => !owned.has(k));
  const extras = Object.keys(editor.pipeline.extras);
  const current = active ?? tabs[0]?.yaml_key ?? OTHER;
  const stage = tabs.find((s) => s.yaml_key === current);
  const pick = (key: string) => editor.select({ kind: "general", stageKey: key } satisfies Selection);
  const errCount = (key: string) => editor.errors.filter((e) => isUnder(e.path, key)).length;

  return (
    <div className="insp">
      <InspectorHead title="General settings" sub="Sections without a lane" />
      <div className="tabs" role="tablist" aria-label="General settings">
        {tabs.map((s) => (
          <button type="button" role="tab" key={s.yaml_key} aria-selected={s.yaml_key === current} className="tabs__tab" onClick={() => pick(s.yaml_key)}>
            {s.title}
            {errCount(s.yaml_key) ? <span className="badge badge--bad">{errCount(s.yaml_key)}</span> : null}
          </button>
        ))}
        {otherKeys.length || extras.length ? (
          <button type="button" role="tab" aria-selected={current === OTHER} className="tabs__tab" onClick={() => pick(OTHER)}>
            Other keys
          </button>
        ) : null}
      </div>
      <div role="tabpanel" className="tabs__panel">
        {stage ? (
          <SectionBody catalog={catalog} stage={stage} />
        ) : (
          <>
            {otherKeys.length ? <h3 className="insp__section">Top-level keys</h3> : null}
            {otherKeys.map((k) => (
              <div key={k}>{fieldControls(rawField(k, editor.pipeline.config[k]), { value: editor.pipeline.config[k], onChange: (v) => editor.setOption([k], v), path: [k], errors: editor.errors })}</div>
            ))}
            {extras.length ? (
              <>
                <h3 className="insp__section">Extras</h3>
                <p className="insp__note">
                  <Icon name="info" size={14} /> Unknown to this server's schema. They are kept and saved as they are, but the CLI will reject them until they are removed.
                </p>
                {extras.map((k) => (
                  <div className="unknown-keys__row" key={k}>
                    {fieldControls(rawField(k, editor.pipeline.extras[k]), { value: editor.pipeline.extras[k], onChange: (v) => editor.setExtra(k, v), path: [k], errors: editor.errors })}
                    <button type="button" className="icon-btn" aria-label={`Remove extra ${k}`} title="Remove" onClick={() => editor.setExtra(k, undefined)}>
                      <Icon name="trash" />
                    </button>
                  </div>
                ))}
              </>
            ) : null}
          </>
        )}
      </div>
    </div>
  );
}
