// Hash routes (`#/runs/<run id>`, `#/runs/<run id>/vs/<run id>`): run folder
// names can contain dots, and a path route whose last segment has one 404s
// on a hard reload. Ids are always URI-encoded, so they never contain "/".
import { useCallback, useEffect, useState } from "react";

export type Route = { view: "editor" } | { view: "runs"; runId: string | null; compareId?: string | null };

function decode(part: string | undefined): string | null {
  if (!part) return null;
  try {
    return decodeURIComponent(part);
  } catch {
    return null;
  }
}

export function parseHash(hash: string): Route {
  const parts = hash.replace(/^#\/?/, "").split("/");
  if (parts[0] !== "runs") return { view: "editor" };
  const runId = decode(parts[1]);
  const compareId = runId && parts[2] === "vs" ? decode(parts[3]) : null;
  return { view: "runs", runId, compareId };
}

export function routeHash(route: Route): string {
  if (route.view === "editor") return "#/";
  if (!route.runId) return "#/runs";
  const base = `#/runs/${encodeURIComponent(route.runId)}`;
  return route.compareId ? `${base}/vs/${encodeURIComponent(route.compareId)}` : base;
}

export function useHashRoute(): [Route, (route: Route) => void] {
  const [route, setRoute] = useState<Route>(() => parseHash(window.location.hash));
  useEffect(() => {
    const onChange = () => setRoute(parseHash(window.location.hash));
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  const navigate = useCallback((next: Route) => {
    const hash = routeHash(next);
    if (window.location.hash !== hash) window.location.hash = hash;
    setRoute(next);
  }, []);
  return [route, navigate];
}
