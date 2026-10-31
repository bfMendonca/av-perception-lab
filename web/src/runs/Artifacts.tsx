// Every file of the run folder as the server lists it: name, type, size,
// Open/Download, and an inline view when one matches (artifactViews).
import { useState } from "react";
import type { Artifact, RunsApi } from "../api/runs";
import { Icon } from "../ui/icons";
import { artifactViewFor } from "./artifactViews";
import { formatBytes } from "./format";

function iconFor(a: Artifact): string {
  if (a.media_type.startsWith("video/")) return "film";
  if (a.media_type.startsWith("image/")) return "spark";
  return "file";
}

export function Artifacts({ artifacts, api }: { artifacts: Artifact[]; api: RunsApi }) {
  const [open, setOpen] = useState<string | null>(null);
  if (!artifacts.length) return <p className="muted">The run folder has no files yet.</p>;
  return (
    <ul className="artifacts">
      {artifacts.map((a) => {
        const View = artifactViewFor(a);
        const expanded = open === a.name && View;
        return (
          <li key={a.name} className={`artifact${expanded ? " artifact--open" : ""}`}>
            <div className="artifact__row">
              <Icon name={iconFor(a)} size={15} />
              {View ? (
                <button type="button" className="artifact__name" aria-expanded={Boolean(expanded)} onClick={() => setOpen(expanded ? null : a.name)}>
                  <Icon name={expanded ? "chevron-down" : "chevron"} size={12} />
                  {a.name}
                </button>
              ) : (
                <span className="artifact__name artifact__name--plain">{a.name}</span>
              )}
              <span className="artifact__type">{a.media_type}</span>
              <span className="artifact__size">{formatBytes(a.size)}</span>
              <a className="icon-btn" href={a.url} target="_blank" rel="noreferrer" aria-label={`Open ${a.name}`} title="Open">
                <Icon name="external" size={14} />
              </a>
              <a className="icon-btn" href={a.url} download={a.name.split("/").pop()} aria-label={`Download ${a.name}`} title="Download">
                <Icon name="save" size={14} />
              </a>
            </div>
            {expanded ? (
              <div className="artifact__view">
                <View artifact={a} api={api} />
              </div>
            ) : null}
          </li>
        );
      })}
    </ul>
  );
}
