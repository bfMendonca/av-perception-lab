// Checkbox dropdown of `render.semantic_classes`: one group per semantic map
// (its classes come from POST /api/classes, keyed by the map's spec), entries
// `<map>.<class>`. Every box checked = the field is absent (all classes); the
// control writes an allowlist only when something is unchecked. The map specs
// come from the config the app puts in StageContext (CONFIG_CONTEXT_KEY).
import { useContext, useEffect, useId, useRef, useState, type ReactElement } from "react";
import { cachedClasses, classesKey, fetchClasses, type DeclaredClasses } from "../api/classes";
import { Icon } from "../ui/icons";
import { isRecord } from "./paths";
import { CONFIG_CONTEXT_KEY, registerFieldControl, StageContext, type FieldControlProps } from "./registry";

const KIND = "semantic";
/** The renderer never draws it (render.SEMANTIC_BACKGROUND). */
const BACKGROUND = "background";
/** Wait for a spec to stop changing (typing a path) before asking the server. */
const DEBOUNCE_MS = 250;
const NOT_KNOWN = "classes appear after the first load";

interface MapSpec {
  name: string;
  spec: Record<string, unknown>;
}

function semanticMaps(config: unknown): MapSpec[] {
  const specs = isRecord(config) ? config[KIND] : undefined;
  if (!Array.isArray(specs)) return [];
  return specs.filter((s): s is Record<string, unknown> => isRecord(s) && typeof s.name === "string" && s.name !== "" && typeof s.type === "string").map((spec) => ({ name: spec.name as string, spec }));
}

/** null = every class (the field is unset). */
function allowlist(value: unknown): string[] | null {
  if (value === null || value === undefined) return null;
  if (typeof value === "string") return [value];
  return Array.isArray(value) ? value.map(String) : null;
}

const mapOf = (entry: string) => entry.split(".")[0];
const classOf = (entry: string) => entry.slice(entry.indexOf(".") + 1);

/** Declared classes per map spec; undefined while the answer is pending. */
function useDeclared(maps: MapSpec[]): (spec: unknown) => DeclaredClasses | undefined {
  const [local, setLocal] = useState<Record<string, DeclaredClasses>>({});
  const latest = useRef(maps);
  latest.current = maps;
  const keys = maps.map((m) => classesKey(KIND, m.spec)).join("\n");
  useEffect(() => {
    const missing = latest.current.filter((m) => !cachedClasses(KIND, m.spec));
    if (!missing.length) return;
    let alive = true;
    const timer = setTimeout(() => {
      for (const m of missing) {
        void fetchClasses(KIND, m.spec).then((result) => {
          if (alive) setLocal((prev) => ({ ...prev, [classesKey(KIND, m.spec)]: result }));
        });
      }
    }, DEBOUNCE_MS);
    return () => {
      alive = false;
      clearTimeout(timer);
    };
  }, [keys]);
  return (spec) => cachedClasses(KIND, spec) ?? local[classesKey(KIND, spec)];
}

export function ClassSelect({ value, onChange, id, errors }: FieldControlProps): ReactElement {
  const maps = semanticMaps(useContext(StageContext)[CONFIG_CONTEXT_KEY]);
  const declared = useDeclared(maps);
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const button = useRef<HTMLButtonElement>(null);
  const panelId = `${id}-panel`;
  const noteId = useId();

  const list = allowlist(value);
  const groups = maps.map((m) => {
    const names = declared(m.spec)?.names;
    return { ...m, loading: declared(m.spec) === undefined, names: names ?? null, entries: (names ?? []).filter((n) => n !== BACKGROUND).map((n) => `${m.name}.${n}`) };
  });
  const known = groups.flatMap((g) => g.entries);
  const knownSet = new Set(known);
  const isBackground = (entry: string) => classOf(entry) === BACKGROUND;
  // Entries of the YAML no fetched list has (a map still loading, a class the
  // server does not declare): kept and shown, never dropped silently.
  const extras = (list ?? []).filter((e) => !knownSet.has(e) && !isBackground(e));
  const selected = new Set(list ?? [...known, ...extras]);

  const write = (next: Set<string>) => {
    if (known.length > 0 && known.every((e) => next.has(e))) {
      onChange(undefined); // all again: the key leaves the YAML
      return;
    }
    onChange([...known, ...extras].filter((e) => next.has(e)));
  };
  const toggle = (entry: string) => {
    const next = new Set(selected);
    if (next.has(entry)) next.delete(entry);
    else next.add(entry);
    write(next);
  };

  const q = query.trim().toLowerCase();
  const matches = (entry: string) => !q || entry.toLowerCase().includes(q);
  const summary = list === null ? "All classes" : list.length === 0 ? "No classes" : known.length ? `${known.filter((e) => selected.has(e)).length} of ${known.length} classes${extras.length ? ` + ${extras.length} kept` : ""}` : `${list.length} listed classes`;

  const row = (entry: string, note?: string) => (
    <label key={entry} className="classes__item">
      <input type="checkbox" aria-label={entry} checked={selected.has(entry)} onChange={() => toggle(entry)} />
      <span>{classOf(entry)}</span>
      {note ? <span className="classes__note">{note}</span> : null}
    </label>
  );

  return (
    <div className="classes">
      <button
        ref={button}
        type="button"
        id={id}
        className="input classes__toggle"
        aria-expanded={open}
        aria-controls={panelId}
        aria-invalid={errors.length ? true : undefined}
        onClick={() => setOpen(!open)}
      >
        <span>{summary}</span>
        <Icon name="chevron" className={`chev${open ? " chev--open" : ""}`} />
      </button>
      {open ? (
        <div
          className="classes__panel"
          id={panelId}
          onKeyDown={(e) => {
            if (e.key === "Escape") {
              setOpen(false);
              button.current?.focus();
            }
          }}
        >
          {maps.length === 0 ? (
            <p className="classes__empty">No semantic segmenter in the pipeline.</p>
          ) : (
            <>
              <div className="classes__bar">
                <div className="classes__search">
                  <Icon name="search" className="text__icon" />
                  <input type="search" className="input" aria-label="Filter classes" placeholder="Filter classes" value={query} spellCheck={false} onChange={(e) => setQuery(e.target.value)} />
                </div>
                <button type="button" className="btn btn--ghost btn--xs" onClick={() => onChange(undefined)}>
                  Select all
                </button>
                <button type="button" className="btn btn--ghost btn--xs" onClick={() => onChange([])}>
                  Select none
                </button>
              </div>
              {groups.map((g) => {
                const mine = extras.filter((e) => mapOf(e) === g.name);
                const entries = g.entries.filter(matches);
                const kept = mine.filter(matches);
                const background = g.names?.includes(BACKGROUND) && matches(`${g.name}.${BACKGROUND}`);
                if (q && !entries.length && !kept.length && !background) return null;
                return (
                  <fieldset key={g.name} className="classes__group">
                    <legend>{g.name}</legend>
                    {g.loading ? <p className="classes__empty">Loading classes…</p> : g.names === null ? <p className="classes__empty">{NOT_KNOWN}</p> : null}
                    {g.names === null && !g.loading && list !== null && !mine.length ? <p className="classes__empty">A class list is set: this map draws none of its classes.</p> : null}
                    {entries.map((e) => row(e))}
                    {background ? (
                      <label className="classes__item classes__item--off">
                        <input type="checkbox" aria-label={`${g.name}.${BACKGROUND}`} checked={false} disabled aria-describedby={`${noteId}-${g.name}`} onChange={() => {}} />
                        <span>{BACKGROUND}</span>
                        <span className="classes__note" id={`${noteId}-${g.name}`}>
                          never drawn
                        </span>
                      </label>
                    ) : null}
                    {kept.map((e) => row(e, g.names === null ? "from the YAML" : "not declared by the model"))}
                  </fieldset>
                );
              })}
              {extras.some((e) => !maps.some((m) => m.name === mapOf(e))) ? (
                <fieldset className="classes__group">
                  <legend>Other entries</legend>
                  {extras.filter((e) => !maps.some((m) => m.name === mapOf(e)) && matches(e)).map((e) => row(e, "no such map"))}
                </fieldset>
              ) : null}
            </>
          )}
        </div>
      ) : null}
    </div>
  );
}

registerFieldControl((f) => f.name === "semantic_classes", ClassSelect);
