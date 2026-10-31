// Hash routes (`#/runs/<run id>`): run folder names can contain dots, and a
// path route whose last segment has one 404s on a hard reload.
import { useCallback, useEffect, useState } from "react";

export type Route = { view: "editor" } | { view: "runs"; runId: string | null };

export function parseHash(hash: string): Route {
  const parts = hash.replace(/^#\/?/, "").split("/");
  if (parts[0] !== "runs") return { view: "editor" };
  let runId: string | null = null;
  if (parts[1]) {
    try {
      runId = decodeURIComponent(parts.slice(1).join("/"));
    } catch {
      runId = null;
    }
  }
  return { view: "runs", runId };
}

export function routeHash(route: Route): string {
  if (route.view === "editor") return "#/";
  return route.runId ? `#/runs/${encodeURIComponent(route.runId)}` : "#/runs";
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
