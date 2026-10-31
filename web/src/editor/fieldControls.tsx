// Form controls generated from the options schema. `Field.type` picks the
// control; an unknown type (or anything without a dedicated control) falls
// back to a raw YAML editor, and `registerFieldControl` overrides any of them.
import { useEffect, useId, useState, type CSSProperties, type ReactElement, type ReactNode } from "react";
import YAML from "yaml";
import type { Field, TypeSchema, ValidationError } from "../api/types";
import { defaultsFromFields, fieldLabel } from "./catalog";
import { deepEqual, isRecord, isUnder, toDotted, type Path } from "./paths";
import { customFieldControl, type FieldControlProps } from "./registry";
import { Icon } from "../ui/icons";

export interface ControlContext {
  value: unknown;
  onChange: (value: unknown) => void;
  path: Path;
  errors?: ValidationError[];
}

const SCALARS = new Set(["bool", "int", "float", "str", "path", "enum"]);
/** Controls that show the errors of their whole subtree (they have no child rows). */
const errorsAt = (errors: ValidationError[], dotted: string, subtree: boolean) =>
  errors.filter((e) => (subtree ? isUnder(e.path, dotted) : e.path === dotted));

/** The control of one field, with its label, help, reset and server errors. */
export function fieldControls(field: Field, ctx: ControlContext): ReactElement {
  return <FieldRow key={toDotted(ctx.path)} field={field} {...ctx} />;
}

function FieldRow({ field, value, onChange, path, errors = [] }: { field: Field } & ControlContext) {
  const id = useId();
  const dotted = toDotted(path);
  const custom = customFieldControl(field);
  const shown = value === undefined ? field.default : value;
  const isGroup = !custom && field.type === "object" && Array.isArray(field.fields) && isRecord(shown);
  const mine = errorsAt(errors, dotted, !isGroup);
  const props: FieldControlProps = { field, value: shown, onChange, path, errors: mine, id };
  const Control = custom ?? defaultControl(field);
  const label = fieldLabel(field);
  const changed = value !== undefined && !deepEqual(value, field.default) && !field.required;
  const errorId = `${id}-errors`;

  if (isGroup && !custom) {
    return (
      <div className="field field--group" data-testid={`field:${dotted}`}>
        <ObjectGroup {...props} allErrors={errors} label={label} onReset={changed ? () => onChange(structuredClone(field.default)) : undefined} />
        <FieldErrors id={errorId} errors={mine} />
      </div>
    );
  }

  return (
    <div className={`field${mine.length ? " field--invalid" : ""}`} data-testid={`field:${dotted}`}>
      <div className="field__head">
        <label className="field__label" htmlFor={id}>
          {label}
          {field.hint?.unit ? <span className="field__unit">{String(field.hint.unit)}</span> : null}
          {field.required ? <span className="field__req" title="Required">*</span> : null}
        </label>
        {label !== field.name ? <code className="field__key">{field.name}</code> : null}
        {changed ? (
          <button type="button" className="icon-btn icon-btn--xs" title={`Reset to default (${YAML.stringify(field.default).trim()})`} aria-label={`Reset ${label} to default`} onClick={() => onChange(structuredClone(field.default))}>
            <Icon name="reset" />
          </button>
        ) : null}
      </div>
      <div className="field__control" aria-describedby={mine.length ? errorId : undefined}>
        <Control {...props} />
      </div>
      {typeof field.hint?.help === "string" ? <p className="field__help">{field.hint.help}</p> : null}
      <FieldErrors id={errorId} errors={mine} />
    </div>
  );
}

function FieldErrors({ id, errors }: { id: string; errors: ValidationError[] }) {
  if (!errors.length) return null;
  return (
    <ul className="field__errors" id={id}>
      {errors.map((e) => (
        <li key={`${e.path}:${e.message}`}>
          <Icon name="alert" />
          <span>{e.message}</span>
        </li>
      ))}
    </ul>
  );
}

function defaultControl(schema: TypeSchema): (p: FieldControlProps) => ReactElement {
  switch (schema.type) {
    case "bool":
      return SwitchControl;
    case "int":
    case "float":
      return hasRange((schema as Field).hint) ? SliderControl : NumberControl;
    case "enum":
      return Array.isArray(schema.choices) ? SelectControl : RawYamlControl;
    case "str":
    case "path":
      return TextControl;
    case "list":
      return schema.items && SCALARS.has(String(schema.items.type)) ? ListControl : RawYamlControl;
    case "object":
      return Array.isArray(schema.fields) ? ObjectControl : RawYamlControl;
    default:
      return RawYamlControl; // map, any, and every type this UI does not know
  }
}

const hasRange = (hint: Field["hint"]) => typeof hint?.min === "number" && typeof hint?.max === "number";
const invalid = (p: FieldControlProps) => (p.errors.length ? true : undefined);

function SwitchControl({ value, onChange, id, field, errors }: FieldControlProps) {
  const on = value === true;
  return (
    <button
      type="button"
      role="switch"
      id={id}
      aria-checked={on}
      aria-invalid={errors.length ? true : undefined}
      className={`switch${on ? " switch--on" : ""}`}
      onClick={() => onChange(!on)}
      title={field.name}
    >
      <span className="switch__thumb" />
      <span className="switch__text">{value === null ? "unset" : on ? "On" : "Off"}</span>
    </button>
  );
}

function parseNumber(text: string, integer: boolean): number | null | "invalid" {
  const t = text.trim();
  if (t === "") return null;
  const n = Number(t);
  if (!Number.isFinite(n) || (integer && !Number.isInteger(n))) return "invalid";
  return n;
}

/** A number box keeping the user's draft text while it does not parse. */
function NumberBox({ value, onChange, id, field, errors, compact }: FieldControlProps & { compact?: boolean }) {
  const integer = field.type === "int";
  const [draft, setDraft] = useState(numberText(value));
  const [bad, setBad] = useState<string | null>(null);
  useEffect(() => {
    setDraft((d) => (parseNumber(d, integer) === value ? d : numberText(value)));
    setBad(null);
  }, [value, integer]);
  return (
    <span className={`num${compact ? " num--compact" : ""}`}>
      <input
        id={compact ? undefined : id}
        aria-label={compact ? `${fieldLabel(field)} value` : undefined}
        className="input input--num"
        inputMode={integer ? "numeric" : "decimal"}
        value={draft}
        placeholder={field.nullable ? "none" : undefined}
        aria-invalid={errors.length || bad ? true : undefined}
        onChange={(e) => {
          setDraft(e.target.value);
          const n = parseNumber(e.target.value, integer);
          if (n === "invalid") setBad(integer ? "Enter a whole number" : "Enter a number");
          else if (n === null && !field.nullable) setBad("A value is required");
          else {
            setBad(null);
            onChange(n);
          }
        }}
      />
      {bad && !compact ? <span className="field__local-error" role="alert">{bad}</span> : null}
    </span>
  );
}

const numberText = (v: unknown) => (typeof v === "number" ? String(v) : "");

function NumberControl(props: FieldControlProps) {
  return <NumberBox {...props} />;
}

function SliderControl(props: FieldControlProps) {
  const { value, onChange, id, field } = props;
  const hint = field.hint ?? {};
  const min = Number(hint.min);
  const max = Number(hint.max);
  const step = typeof hint.step === "number" ? hint.step : field.type === "int" ? 1 : (max - min) / 100;
  const n = typeof value === "number" ? value : min;
  const pct = max > min ? Math.min(100, Math.max(0, ((n - min) / (max - min)) * 100)) : 0;
  return (
    <div className="slider">
      <input
        type="range"
        id={id}
        className="slider__range"
        min={min}
        max={max}
        step={step}
        value={n}
        aria-invalid={invalid(props)}
        aria-valuetext={value === null ? "none" : undefined}
        style={{ "--pct": `${pct}%` } as CSSProperties}
        onChange={(e) => onChange(field.type === "int" ? Math.round(Number(e.target.value)) : Number(e.target.value))}
      />
      <NumberBox {...props} compact />
    </div>
  );
}

function SelectControl(props: FieldControlProps) {
  const { value, onChange, id, field } = props;
  const choices = (field.choices ?? []) as unknown[];
  const NONE = "\u0000none";
  const current = choices.find((c) => deepEqual(c, value));
  const unknown = value !== null && value !== undefined && current === undefined;
  return (
    <div className="select">
      <select
        id={id}
        className="input select__el"
        value={current !== undefined ? String(current) : value === null || value === undefined ? NONE : String(value)}
        aria-invalid={invalid(props)}
        onChange={(e) => {
          const v = e.target.value;
          if (v === NONE) onChange(null);
          else {
            const choice = choices.find((c) => String(c) === v);
            if (choice !== undefined) onChange(choice);
          }
        }}
      >
        {field.nullable || value === null || value === undefined ? <option value={NONE}>none</option> : null}
        {unknown ? <option value={String(value)}>{String(value)} (not a choice)</option> : null}
        {choices.map((c) => (
          <option key={String(c)} value={String(c)}>
            {String(c)}
          </option>
        ))}
      </select>
      <Icon name="chevron" className="select__icon" />
    </div>
  );
}

function TextControl(props: FieldControlProps) {
  const { value, onChange, id, field } = props;
  const isPath = field.type === "path";
  return (
    <div className={`text${isPath ? " text--path" : ""}`}>
      {isPath ? <Icon name="folder" className="text__icon" /> : null}
      <input
        id={id}
        className={`input${isPath ? " input--mono" : ""}`}
        value={typeof value === "string" ? value : value === null || value === undefined ? "" : String(value)}
        placeholder={field.nullable ? "none" : isPath ? "path on the server" : undefined}
        spellCheck={false}
        aria-invalid={invalid(props)}
        onChange={(e) => onChange(e.target.value === "" && field.nullable ? null : e.target.value)}
      />
    </div>
  );
}

function coerceItem(schema: TypeSchema, text: string): unknown {
  if (schema.type === "int" || schema.type === "float") {
    const n = Number(text);
    return text.trim() !== "" && Number.isFinite(n) ? n : text;
  }
  if (schema.type === "bool") return text === "true";
  return text;
}

function ListControl(props: FieldControlProps) {
  const { value, onChange, id, field } = props;
  const items = field.items as TypeSchema;
  const fixed = typeof field.length === "number";
  if (value === null || value === undefined) {
    return (
      <div className="unset">
        <span>Not set</span>
        <button type="button" className="btn btn--ghost btn--sm" id={id} onClick={() => onChange([])}>
          <Icon name="plus" /> Set list
        </button>
      </div>
    );
  }
  if (!Array.isArray(value)) return <RawYamlControl {...props} />;
  return (
    <div className="list" id={id} role="group" aria-label={fieldLabel(field)}>
      {value.map((item, i) => (
        <div className="list__row" key={i}>
          {items.type === "enum" && Array.isArray(items.choices) ? (
            <select className="input" aria-label={`${fieldLabel(field)} item ${i + 1}`} value={String(item)} onChange={(e) => onChange(value.map((v, j) => (j === i ? (items.choices as unknown[]).find((c) => String(c) === e.target.value) : v)))}>
              {(items.choices as unknown[]).map((c) => (
                <option key={String(c)} value={String(c)}>
                  {String(c)}
                </option>
              ))}
            </select>
          ) : (
            <input className={`input${items.type === "path" ? " input--mono" : ""}`} aria-label={`${fieldLabel(field)} item ${i + 1}`} value={item === null || item === undefined ? "" : String(item)} spellCheck={false} onChange={(e) => onChange(value.map((v, j) => (j === i ? coerceItem(items, e.target.value) : v)))} />
          )}
          {fixed ? null : (
            <button type="button" className="icon-btn" aria-label={`Remove item ${i + 1}`} onClick={() => onChange(value.filter((_, j) => j !== i))}>
              <Icon name="x" />
            </button>
          )}
        </div>
      ))}
      <div className="list__foot">
        {fixed ? null : (
          <button type="button" className="btn btn--ghost btn--sm" onClick={() => onChange([...value, items.type === "int" || items.type === "float" ? 0 : ""])}>
            <Icon name="plus" /> Add item
          </button>
        )}
        {field.nullable ? (
          <button type="button" className="btn btn--ghost btn--sm" onClick={() => onChange(null)}>
            Unset
          </button>
        ) : null}
      </div>
    </div>
  );
}

/** A nullable object that is null: offer to enable it with its defaults. */
function ObjectControl(props: FieldControlProps) {
  const { onChange, id, field } = props;
  return (
    <div className="unset">
      <span>Not set</span>
      <button type="button" id={id} className="btn btn--ghost btn--sm" onClick={() => onChange(isRecord(field.default) ? structuredClone(field.default) : defaultsFromFields(field.fields ?? []))}>
        <Icon name="plus" /> Enable
      </button>
    </div>
  );
}

function ObjectGroup(props: FieldControlProps & { allErrors: ValidationError[]; label: string; onReset?: () => void }) {
  const { value, onChange, path, field, allErrors, label, onReset } = props;
  const [open, setOpen] = useState(true);
  const obj = isRecord(value) ? value : {};
  const count = allErrors.filter((e) => isUnder(e.path, toDotted(path)) && e.path !== toDotted(path)).length;
  return (
    <section className="group">
      <header className="group__head">
        <button type="button" className="group__toggle" aria-expanded={open} onClick={() => setOpen(!open)}>
          <Icon name="chevron" className={`chev${open ? " chev--open" : ""}`} />
          <span>{label}</span>
          {label !== field.name ? <code className="field__key">{field.name}</code> : null}
          {count ? <span className="badge badge--bad">{count}</span> : null}
        </button>
        {onReset ? (
          <button type="button" className="icon-btn icon-btn--xs" aria-label={`Reset ${label} to default`} title="Reset to default" onClick={onReset}>
            <Icon name="reset" />
          </button>
        ) : null}
        {field.nullable ? (
          <button type="button" className="btn btn--ghost btn--xs" onClick={() => onChange(null)}>
            Unset
          </button>
        ) : null}
      </header>
      {typeof field.hint?.help === "string" ? <p className="field__help">{field.hint.help}</p> : null}
      {open ? (
        <div className="group__body">
          <FieldList fields={field.fields ?? []} value={obj} basePath={path} errors={allErrors} onChange={(name, v) => onChange(setKey(obj, name, v))} />
        </div>
      ) : null}
    </section>
  );
}

function setKey(obj: Record<string, unknown>, key: string, v: unknown) {
  const out = { ...obj };
  if (v === undefined) delete out[key];
  else out[key] = v;
  return out;
}

const toYaml = (v: unknown) => (v === undefined ? "" : YAML.stringify(v, { lineWidth: 0 }).replace(/\n$/, ""));

/** Fallback for any field: edit the value as YAML; committed on blur when it parses. */
export function RawYamlControl(props: FieldControlProps) {
  const { value, onChange, id, field } = props;
  const [text, setText] = useState(toYaml(value));
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    setText(toYaml(value));
    setError(null);
  }, [value]);
  const commit = () => {
    if (text === toYaml(value)) {
      setError(null);
      return;
    }
    try {
      const parsed = text.trim() === "" ? null : YAML.parse(text);
      setError(null);
      onChange(parsed);
    } catch (exc) {
      setError(`Invalid YAML: ${exc instanceof Error ? exc.message.split("\n")[0] : String(exc)}`);
    }
  };
  const rows = Math.min(10, Math.max(2, text.split("\n").length));
  return (
    <div className="raw">
      <textarea
        id={id}
        className="input input--mono raw__text"
        value={text}
        rows={rows}
        spellCheck={false}
        aria-invalid={error || props.errors.length ? true : undefined}
        onChange={(e) => setText(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) commit();
        }}
      />
      <span className="raw__badge" title={`Type "${String(field.type)}" is edited as YAML`}>
        YAML
      </span>
      {error ? (
        <span className="field__local-error" role="alert">
          {error}
        </span>
      ) : null}
    </div>
  );
}

export interface FieldListProps {
  fields: readonly Field[];
  /** The object holding the fields' values. */
  value: unknown;
  basePath: Path;
  /** Called with the field name and its new value (`undefined` = remove). */
  onChange: (name: string, value: unknown) => void;
  errors?: ValidationError[];
}

/** The fields of an object, "Advanced" ones in a group collapsed by default. */
export function FieldList({ fields, value, basePath, onChange, errors = [] }: FieldListProps) {
  const obj = isRecord(value) ? value : {};
  const basic = fields.filter((f) => !f.hint?.advanced);
  const advanced = fields.filter((f) => f.hint?.advanced);
  const row = (f: Field) => fieldControls(f, { value: obj[f.name], onChange: (v) => onChange(f.name, v), path: [...basePath, f.name], errors });
  return (
    <div className="fields">
      {basic.map(row)}
      {advanced.length ? (
        <Collapsible title="Advanced" count={advanced.length} badErrors={advanced.some((f) => errors.some((e) => isUnder(e.path, toDotted([...basePath, f.name]))))}>
          {advanced.map(row)}
        </Collapsible>
      ) : null}
    </div>
  );
}

export function Collapsible({ title, count, children, badErrors, defaultOpen = false }: { title: string; count?: number; children: ReactNode; badErrors?: boolean; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(defaultOpen);
  const isOpen = open || Boolean(badErrors);
  return (
    <div className={`collapsible${isOpen ? " collapsible--open" : ""}`}>
      <button type="button" className="collapsible__toggle" aria-expanded={isOpen} onClick={() => setOpen(!isOpen)}>
        <Icon name="chevron" className={`chev${isOpen ? " chev--open" : ""}`} />
        <span>{title}</span>
        {count !== undefined ? <span className="collapsible__count">{count}</span> : null}
        {badErrors ? <span className="badge badge--bad">!</span> : null}
      </button>
      {isOpen ? <div className="collapsible__body">{children}</div> : null}
    </div>
  );
}
