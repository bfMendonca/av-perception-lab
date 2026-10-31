// Stage lanes in catalog order. Each stage renders through a registered view
// when there is one, otherwise through the generic lane: adapter stages get a
// card per spec, section stages a single card, anything else a raw card.
import { useEffect, useRef, useState } from "react";
import type { Catalog, StageInfo, ValidationError } from "../api/types";
import { Icon } from "../ui/icons";
import { laneUnits, moduleInfo, moduleTypes, sectionFields, slotPath, slotsOf, summarize } from "./catalog";
import { EmptySlot, ModuleCard } from "./ModuleCard";
import { isRecord, isUnder, toDotted } from "./paths";
import { stageViewFor } from "./registry";
import { useEditor, type Selection } from "./yamlState";

export function Lanes() {
  const { catalog, pipeline, errors, selection, select } = useEditor();
  if (!catalog) return null;
  const stage = (s: StageInfo, nested: boolean) => {
    const View = stageViewFor(s.yaml_key);
    const props = {
      stage: s,
      value: pipeline.config[s.yaml_key],
      catalog,
      errors: errors.filter((e) => isUnder(e.path, s.yaml_key)),
      selection,
      select,
    };
    return View ? <View key={s.yaml_key} {...props} /> : <GenericLane key={s.yaml_key} {...props} nested={nested} />;
  };
  return (
    <div className="lanes" role="list" aria-label="Pipeline stages">
      {laneUnits(catalog).map((unit, i) => (
        <div className="lanes__item" role="listitem" key={unit.key}>
          {i > 0 ? (
            <div className="lanes__arrow" aria-hidden="true">
              <svg width="22" height="10" viewBox="0 0 22 10">
                <path d="M0 5h19M15 1l4 4-4 4" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </div>
          ) : null}
          {unit.group !== null ? (
            <section className="lane lane--group" aria-label={unit.group}>
              <header className="lane__head">
                <h2 className="lane__title">{unit.group}</h2>
              </header>
              <div className="lane__body">{unit.stages.map((s) => stage(s, true))}</div>
            </section>
          ) : (
            stage(unit.stages[0], false)
          )}
        </div>
      ))}
    </div>
  );
}

const isSelected = (sel: Selection | null, stage: StageInfo, index: number | null) =>
  sel?.kind === "slot" && sel.stageKey === stage.yaml_key && sel.index === index;

interface LaneProps {
  stage: StageInfo;
  value: unknown;
  catalog: Catalog;
  errors: ValidationError[];
  selection: Selection | null;
  select: (s: Selection) => void;
}

/** The generic lane of a stage; `nested` renders it as a part of a group lane. */
export function GenericLane({ stage, value, catalog, errors, selection, select, nested = false }: LaneProps & { nested?: boolean }) {
  const { addModule } = useEditor();
  const slots = slotsOf(stage, value);
  const ownErrors = errors.filter((e) => e.path === stage.yaml_key);
  const canAdd = stage.multiple || slots.length === 0;
  const badShape = stage.multiple ? value !== undefined && value !== null && !Array.isArray(value) : false;

  return (
    <section className={nested ? "lane__part" : "lane"} aria-label={stage.title}>
      {nested ? (
        <h3 className="lane__subtitle">{stage.title}</h3>
      ) : (
        <header className="lane__head">
          <h2 className="lane__title" title={stage.optional ? `${stage.title} (optional)` : stage.title}>
            {stage.title}
          </h2>
        </header>
      )}
      <div className="lane__body">
        {slots.map(({ index, value: spec }) => {
          const dotted = toDotted(slotPath(stage, index));
          const errorCount = errors.filter((e) => isUnder(e.path, dotted)).length;
          const onSelect = () => select({ kind: "slot", stageKey: stage.yaml_key, index });
          if (stage.kind !== null) {
            const type = isRecord(spec) && typeof spec.type === "string" ? spec.type : null;
            const info = moduleInfo(catalog, stage.kind, type);
            return (
              <ModuleCard
                key={dotted}
                title={type ?? "untyped"}
                tag={isRecord(spec) && typeof spec.name === "string" ? spec.name : undefined}
                subtitle={summarize(info?.schema.fields, spec)}
                description={info?.doc}
                tone={info ? "default" : "unknown"}
                selected={isSelected(selection, stage, index)}
                errorCount={errorCount}
                onSelect={onSelect}
              />
            );
          }
          return (
            <ModuleCard
              key={dotted}
              title={stage.title}
              subtitle={summarize(sectionFields(catalog, stage.yaml_key), spec, 2) || (isRecord(spec) ? "defaults" : String(spec))}
              selected={isSelected(selection, stage, index)}
              errorCount={errorCount}
              onSelect={onSelect}
            />
          );
        })}
        {badShape ? (
          <ModuleCard title={stage.title} subtitle="not a list: edit as YAML" tone="unknown" errorCount={errors.length} selected={isSelected(selection, stage, null)} onSelect={() => select({ kind: "slot", stageKey: stage.yaml_key, index: null })} />
        ) : null}
        {slots.length === 0 && !badShape ? <EmptySlot label={stage.optional ? "Off" : "Not configured"} hint={stage.optional ? undefined : "required"} /> : null}
        {canAdd && !badShape ? <AddButton stage={stage} catalog={catalog} onAdd={(t) => addModule(stage.yaml_key, t)} /> : null}
        {ownErrors.map((e, i) => (
          <p className="lane__error" key={i}>
            <Icon name="alert" size={13} /> {e.message}
          </p>
        ))}
      </div>
    </section>
  );
}

/** "+ add": a menu of the stage's module types, or a plain "Enable" for a section. */
function AddButton({ stage, catalog, onAdd }: { stage: StageInfo; catalog: Catalog; onAdd: (type?: string) => void }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const types = moduleTypes(catalog, stage.kind);
  useEffect(() => {
    if (!open) return;
    const close = (e: MouseEvent | KeyboardEvent) => {
      if (e instanceof KeyboardEvent ? e.key === "Escape" : !ref.current?.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", close);
    document.addEventListener("keydown", close);
    return () => {
      document.removeEventListener("mousedown", close);
      document.removeEventListener("keydown", close);
    };
  }, [open]);

  if (stage.kind === null) {
    return (
      <button type="button" className="add" aria-label={`Enable ${stage.title}`} onClick={() => onAdd()}>
        <Icon name="plus" size={14} /> Enable
      </button>
    );
  }
  return (
    <div className="add-wrap" ref={ref}>
      <button
        type="button"
        className="add"
        aria-label={`Add to ${stage.title}`}
        aria-haspopup="menu"
        aria-expanded={open}
        disabled={!types.length}
        title={types.length ? undefined : `No ${stage.kind} module is registered`}
        onClick={() => setOpen(!open)}
      >
        <Icon name="plus" size={14} /> Add
      </button>
      {open ? (
        <div className="menu" role="menu" aria-label={`Add to ${stage.title}`}>
          {types.map((t) => {
            const doc = catalog.modules[stage.kind as string]?.[t]?.doc;
            return (
              <button
                type="button"
                role="menuitem"
                key={t}
                className="menu__item"
                autoFocus={t === types[0]}
                onClick={() => {
                  setOpen(false);
                  onAdd(t);
                }}
              >
                <span className="menu__name">{t}</span>
                {doc ? <span className="menu__doc">{doc}</span> : null}
              </button>
            );
          })}
        </div>
      ) : null}
    </div>
  );
}
