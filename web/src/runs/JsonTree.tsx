// Generic collapsible key/value tree for any JSON (metrics.json, a new sink's
// output): sections the UI has never heard of render like the known ones.
// Children mount only when opened, and long lists page, so a large file
// (metrics.json with hundreds of tracks) stays cheap.
import { useState } from "react";

const PAGE = 100;

function isObject(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function scalar(v: unknown): { text: string; kind: string } {
  if (v === null || v === undefined) return { text: "null", kind: "null" };
  if (typeof v === "number") return { text: Number.isInteger(v) ? String(v) : String(Number(v.toPrecision(6))), kind: "number" };
  if (typeof v === "boolean") return { text: String(v), kind: "bool" };
  return { text: JSON.stringify(v), kind: "string" };
}

function summary(v: unknown): string {
  if (Array.isArray(v)) return `[${v.length} item${v.length === 1 ? "" : "s"}]`;
  if (isObject(v)) {
    const n = Object.keys(v).length;
    return `{${n} key${n === 1 ? "" : "s"}}`;
  }
  return "";
}

function Node({ name, value, depth, defaultOpen }: { name: string; value: unknown; depth: number; defaultOpen: boolean }) {
  const [open, setOpen] = useState(defaultOpen);
  const [shown, setShown] = useState(PAGE);
  if (!Array.isArray(value) && !isObject(value)) {
    const s = scalar(value);
    return (
      <li className="jtree__leaf">
        <span className="jtree__key">{name}</span>
        <span className={`jtree__val jtree__val--${s.kind}`}>{s.text}</span>
      </li>
    );
  }
  const entries: [string, unknown][] = Array.isArray(value) ? value.map((v, i) => [String(i), v]) : Object.entries(value);
  return (
    <li className="jtree__branch">
      <details open={open} onToggle={(e) => setOpen((e.currentTarget as HTMLDetailsElement).open)}>
        <summary>
          <span className="jtree__key">{name}</span>
          <span className="jtree__count">{summary(value)}</span>
        </summary>
        {open ? (
          <ul className="jtree__list">
            {entries.slice(0, shown).map(([k, v]) => (
              <Node key={k} name={k} value={v} depth={depth + 1} defaultOpen={false} />
            ))}
            {entries.length > shown ? (
              <li>
                <button type="button" className="btn btn--xs btn--ghost" onClick={() => setShown(shown + PAGE)}>
                  Show {Math.min(PAGE, entries.length - shown)} more of {entries.length - shown}
                </button>
              </li>
            ) : null}
          </ul>
        ) : null}
      </details>
    </li>
  );
}

export function JsonTree({ value, label = "root", openRoot = true }: { value: unknown; label?: string; openRoot?: boolean }) {
  if (!Array.isArray(value) && !isObject(value)) {
    return (
      <ul className="jtree" aria-label={label}>
        <Node name={label} value={value} depth={0} defaultOpen />
      </ul>
    );
  }
  const entries: [string, unknown][] = Array.isArray(value) ? value.map((v, i) => [String(i), v]) : Object.entries(value);
  return (
    <ul className="jtree" aria-label={label}>
      {openRoot ? entries.map(([k, v]) => <Node key={k} name={k} value={v} depth={0} defaultOpen={false} />) : <Node name={label} value={value} depth={0} defaultOpen={false} />}
    </ul>
  );
}
