// Client of POST /api/classes (src/av_perception_lab/web/classes.py): the
// classes an adapter spec declares, known without loading the model.
import { readError } from "./client";

export interface DeclaredClasses {
  /** null: not knowable before the first load (or the spec was rejected). */
  names: string[] | null;
  source: string | null;
}

export interface ClassesApi {
  /** A rejected spec (422) resolves to `{names: null}`: it means "no list", not a failure. */
  fetch(kind: string, spec: unknown): Promise<DeclaredClasses>;
}

const UNKNOWN: DeclaredClasses = { names: null, source: null };

export function createClassesApi(base = "", fetchImpl: typeof fetch = (...a) => fetch(...a)): ClassesApi {
  return {
    async fetch(kind, spec) {
      const response = await fetchImpl(`${base}/api/classes`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ kind, spec }),
      });
      if (response.status === 422) return UNKNOWN;
      if (!response.ok) throw await readError(response);
      const body = (await response.json()) as Partial<DeclaredClasses>;
      return { names: Array.isArray(body.names) ? body.names.map(String) : null, source: typeof body.source === "string" ? body.source : null };
    },
  };
}

/** Stable text of a JSON value (object keys sorted), the cache key of a spec. */
export function stableKey(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(stableKey).join(",")}]`;
  if (value && typeof value === "object") {
    const obj = value as Record<string, unknown>;
    return `{${Object.keys(obj)
      .sort()
      .filter((k) => obj[k] !== undefined)
      .map((k) => `${JSON.stringify(k)}:${stableKey(obj[k])}`)
      .join(",")}}`;
  }
  return JSON.stringify(value) ?? "null";
}

// One request per distinct (kind, spec) for the page's lifetime once a class
// list is known: settled answers are read synchronously, in-flight ones are
// shared; an "unknown" answer is never cached (the caller decides when to ask
// again).
const settled = new Map<string, DeclaredClasses>();
const pending = new Map<string, Promise<DeclaredClasses>>();
const defaultApi = createClassesApi();

export const classesKey = (kind: string, spec: unknown) => `${kind}:${stableKey(spec)}`;

/** The cached answer for a spec, or undefined when it was not fetched yet. */
export const cachedClasses = (kind: string, spec: unknown): DeclaredClasses | undefined => settled.get(classesKey(kind, spec));

/**
 * The classes `spec` declares, fetched once per distinct spec. A network or
 * server error resolves to "unknown" without being cached, so a later mount
 * may try again.
 */
export function fetchClasses(kind: string, spec: unknown, api: ClassesApi = defaultApi): Promise<DeclaredClasses> {
  const key = classesKey(kind, spec);
  const done = settled.get(key);
  if (done) return Promise.resolve(done);
  let request = pending.get(key);
  if (!request) {
    request = api.fetch(kind, spec).then(
      (result) => {
        // "unknown" is not final (weights may be prepared later): not cached.
        if (result.names !== null) settled.set(key, result);
        return result;
      },
      () => UNKNOWN,
    );
    request.finally(() => pending.delete(key));
    pending.set(key, request);
  }
  return request;
}

/** Forget every answer (tests). */
export function resetClassesCache(): void {
  settled.clear();
  pending.clear();
}
