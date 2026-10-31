// Summary cards of the selected run with signed deltas against the
// comparison run, and the diff of their effective configurations. The
// metrics (labels, units, which direction is better) come from the server.
import { useState } from "react";
import YAML from "yaml";
import type { Delta, DiffEntry, MetricInfo, Summary } from "../api/runs";
import { Icon } from "../ui/icons";
import { deltaView, formatValue } from "./format";

export interface KpiCardsProps {
  metrics: MetricInfo[];
  summary: Summary;
  /** Deltas B - A, by metric key (only metrics both runs have). */
  deltas?: Record<string, Delta> | null;
  comparing?: boolean;
}

export function KpiCards({ metrics, summary, deltas, comparing }: KpiCardsProps) {
  // Metrics neither run has are left out; one the selected run lacks shows n/a.
  const shown = metrics.filter((m) => typeof summary[m.key] === "number" || (deltas && m.key in deltas));
  if (!shown.length) return <p className="muted">No summary values for this run.</p>;
  return (
    <div className="kpis" role="list">
      {shown.map((m) => {
        const d = comparing ? deltaView(m, deltas?.[m.key]) : null;
        return (
          <div className="kpi" role="listitem" key={m.key} title={m.help}>
            <span className="kpi__label">{m.label}</span>
            <strong className="kpi__value">{formatValue(summary[m.key], m.unit)}</strong>
            {d ? (
              <span className={`kpi__delta kpi__delta--${d.tone}`} aria-label={d.label}>
                {d.tone !== "neutral" ? <Icon name={d.tone === "good" ? "check" : "alert"} size={11} /> : null}
                {d.text}
              </span>
            ) : comparing ? (
              <span className="kpi__delta kpi__delta--none">no comparison</span>
            ) : null}
          </div>
        );
      })}
    </div>
  );
}

const show = (v: unknown) => YAML.stringify(v, { lineWidth: 0, collectionStyle: "flow" }).trim();

export function ConfigDiff({ diff, origin }: { diff: DiffEntry[] | null; origin?: { a: string | null; b: string | null } }) {
  const [showServer, setShowServer] = useState(false);
  if (diff === null) {
    const missing = origin && !origin.a ? "The comparison run" : "This run";
    return <p className="muted diff__empty">{missing} recorded no configuration (it predates config.yaml and the configuration in metrics.json).</p>;
  }
  const own = diff.filter((d) => !d.server_set);
  const server = diff.filter((d) => d.server_set);
  const entries = showServer ? diff : own;
  return (
    <div className="cdiff">
      {own.length === 0 ? <p className="muted diff__empty">Same configuration{server.length ? " (apart from the run folder the server picks)" : ""}.</p> : null}
      {entries.length ? (
        <pre className="diff__body cdiff__body" tabIndex={0} aria-label="Configuration changes">
          {entries.map((e) => (
            <span className={`diff__entry${e.server_set ? " cdiff__server" : ""}`} key={e.path}>
              {e.change !== "added" ? <span className="diff__del">{`- ${e.path}: ${show(e.a)}`}</span> : null}
              {e.change !== "removed" ? <span className="diff__add">{`+ ${e.path}: ${show(e.b)}`}</span> : null}
            </span>
          ))}
        </pre>
      ) : null}
      {server.length ? (
        <button type="button" className="btn btn--xs btn--ghost cdiff__toggle" onClick={() => setShowServer(!showServer)}>
          {showServer ? "Hide" : "Show"} {server.length} key{server.length > 1 ? "s" : ""} set by the server ({server.map((d) => d.path).join(", ")})
        </button>
      ) : null}
    </div>
  );
}
