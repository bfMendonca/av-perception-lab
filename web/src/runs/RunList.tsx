// History list: the selected run (accent) and the comparison run (violet);
// each entry can be picked as the comparison of the selected run.
import { useMemo, useState } from "react";
import type { RunRecord } from "../api/video";
import { Icon } from "../ui/icons";
import { formatWhen } from "./format";

export interface RunListProps {
  runs: RunRecord[] | null;
  error: string | null;
  selected: string | null;
  compareId: string | null;
  onSelect: (id: string) => void;
  onCompare: (id: string | null) => void;
  onRefresh: () => void;
  onDelete: (run: RunRecord) => void;
}

/** Runs with a folder (a `job-<id>` entry is a failed job that left none). */
export const hasFiles = (r: RunRecord) => !r.id.startsWith("job-");

export function runTitle(r: RunRecord): string {
  return r.video ?? r.id;
}

export function RunList({ runs, error, selected, compareId, onSelect, onCompare, onRefresh, onDelete }: RunListProps) {
  const [query, setQuery] = useState("");
  const visible = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!runs || !q) return runs;
    return runs.filter((r) => [r.id, r.video ?? "", r.status, ...r.detectors, r.depth ?? ""].some((s) => s.toLowerCase().includes(q)));
  }, [query, runs]);

  return (
    <aside className="runs__list" aria-label="Run history">
      <div className="runs__list-head">
        <h3>History</h3>
        {runs ? <span className="runs__count">{runs.length}</span> : null}
        <button type="button" className="icon-btn" aria-label="Refresh the history" title="Refresh" onClick={onRefresh}>
          <Icon name="reset" size={14} />
        </button>
      </div>
      <label className="runs__search">
        <Icon name="search" size={14} />
        <input className="input" type="search" placeholder="Filter by video, model, status" value={query} onChange={(e) => setQuery(e.target.value)} aria-label="Filter runs" />
      </label>
      {error ? <p className="muted">Could not load the history: {error}</p> : null}
      {runs === null && !error ? (
        <div className="runs__skeleton" aria-busy="true">
          {[0, 1, 2, 3].map((i) => (
            <span className="skel skel--card" key={i} />
          ))}
        </div>
      ) : null}
      {runs?.length === 0 ? <p className="muted runs__empty">No runs yet. Process a video segment from the editor.</p> : null}
      {visible && runs?.length && !visible.length ? <p className="muted runs__empty">No run matches “{query}”.</p> : null}
      <ul>
        {visible?.map((r) => {
          const isSel = r.id === selected;
          const isCmp = r.id === compareId;
          const canCompare = Boolean(selected) && !isSel && hasFiles(r);
          return (
            <li key={r.id} className={`runitem${isSel ? " runitem--on" : ""}${isCmp ? " runitem--cmp" : ""}`}>
              <button type="button" className="runitem__main" aria-current={isSel ? "true" : undefined} onClick={() => onSelect(r.id)} title={r.id}>
                <span className="runitem__name">{runTitle(r)}</span>
                <span className="runitem__meta">
                  <span className={`job__dot job__dot--${r.status}`} aria-hidden="true" /> {r.status} · {formatWhen(r.created)}
                </span>
                <span className="runitem__meta runitem__models">
                  {[r.detectors.join(" + "), r.depth].filter(Boolean).join(" · ") || (r.error ? r.error : "—")}
                  {r.frames ? ` · ${r.frames} frames` : ""}
                </span>
                {isCmp ? <span className="runitem__tag">comparing</span> : null}
              </button>
              {canCompare ? (
                <button
                  type="button"
                  className={`runitem__cmp${isCmp ? " runitem__cmp--on" : ""}`}
                  aria-pressed={isCmp}
                  aria-label={isCmp ? `Stop comparing with ${runTitle(r)}` : `Compare with ${runTitle(r)}`}
                  title={isCmp ? "Stop comparing" : "Compare the selected run with this one"}
                  onClick={() => onCompare(isCmp ? null : r.id)}
                >
                  <Icon name="compare" size={14} />
                </button>
              ) : null}
              <button type="button" className="runitem__cmp runitem__del" aria-label={`Delete ${runTitle(r)}`} title="Delete this run" onClick={() => onDelete(r)}>
                <Icon name="trash" size={14} />
              </button>
            </li>
          );
        })}
      </ul>
    </aside>
  );
}
