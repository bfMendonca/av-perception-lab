// Views of run artifacts, picked by a predicate on the listed file (name,
// media type, role). Files no view matches still get Open/Download, so a new
// sink's output shows up without UI changes; a custom view is one
// `registerArtifactView` call (the latest registration wins).
import { useEffect, useState, type ReactElement } from "react";
import type { Artifact, RunsApi } from "../api/runs";
import { JsonTree } from "./JsonTree";

export interface ArtifactViewProps {
  artifact: Artifact;
  api: RunsApi;
}

export type ArtifactView = (props: ArtifactViewProps) => ReactElement | null;

interface Entry {
  match: (a: Artifact) => boolean;
  view: ArtifactView;
}

const custom: Entry[] = [];

export function registerArtifactView(match: (a: Artifact) => boolean, view: ArtifactView): () => void {
  const entry = { match, view };
  custom.unshift(entry);
  return () => {
    const i = custom.indexOf(entry);
    if (i >= 0) custom.splice(i, 1);
  };
}

const JSON_MAX_BYTES = 8 * 1024 * 1024;

function JsonArtifact({ artifact, api }: ArtifactViewProps) {
  const [state, setState] = useState<{ value?: unknown; error?: string }>({});
  useEffect(() => {
    let live = true;
    api
      .fetchJson(artifact.url)
      .then((value) => live && setState({ value }))
      .catch((exc: unknown) => live && setState({ error: exc instanceof Error ? exc.message : String(exc) }));
    return () => {
      live = false;
    };
  }, [api, artifact.url]);
  if (state.error) return <p className="muted">Could not read {artifact.name}: {state.error}</p>;
  if (!("value" in state)) return <p className="muted">Loading {artifact.name}…</p>;
  return <JsonTree value={state.value} label={artifact.name} />;
}

function ImageArtifact({ artifact }: ArtifactViewProps) {
  return <img className="artifact__img" src={artifact.url} alt={artifact.name} loading="lazy" />;
}

function VideoArtifact({ artifact }: ArtifactViewProps) {
  return <video className="artifact__video" src={artifact.url} controls muted playsInline preload="metadata" />;
}

const builtIn: Entry[] = [
  { match: (a) => a.media_type === "application/json" && a.size <= JSON_MAX_BYTES, view: JsonArtifact },
  { match: (a) => a.media_type.startsWith("image/"), view: ImageArtifact },
  { match: (a) => a.media_type === "video/mp4" || a.media_type === "video/webm", view: VideoArtifact },
];

/** The view of an artifact: a registered one first, then the built-ins; undefined = links only. */
export function artifactViewFor(artifact: Artifact): ArtifactView | undefined {
  return (custom.find((e) => e.match(artifact)) ?? builtIn.find((e) => e.match(artifact)))?.view;
}
