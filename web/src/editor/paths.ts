// Paths into the config tree and the dotted form the server uses in errors
// (`detectors[0].threshold`).

export type PathSeg = string | number;
export type Path = readonly PathSeg[];

export function toDotted(path: Path): string {
  let out = "";
  for (const seg of path) {
    if (typeof seg === "number") out += `[${seg}]`;
    else out += out === "" ? seg : `.${seg}`;
  }
  return out;
}

/** True when `dotted` is `prefix` itself or lies under it. */
export function isUnder(dotted: string, prefix: string): boolean {
  if (prefix === "") return true;
  if (dotted === prefix) return true;
  return dotted.startsWith(prefix) && (dotted[prefix.length] === "." || dotted[prefix.length] === "[");
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function getIn(root: unknown, path: Path): unknown {
  let node = root;
  for (const seg of path) {
    if (Array.isArray(node) && typeof seg === "number") node = node[seg];
    else if (isRecord(node) && typeof seg === "string") node = node[seg];
    else return undefined;
  }
  return node;
}

/** Immutable set; `undefined` removes the key (or list item). Missing parents are created. */
export function setIn(root: unknown, path: Path, value: unknown): unknown {
  if (path.length === 0) return value;
  const [head, ...rest] = path;
  if (typeof head === "number") {
    const list = Array.isArray(root) ? [...root] : [];
    if (rest.length === 0 && value === undefined) {
      list.splice(head, 1);
      return list;
    }
    list[head] = setIn(list[head], rest, value);
    return list;
  }
  const obj: Record<string, unknown> = isRecord(root) ? { ...root } : {};
  if (rest.length === 0 && value === undefined) {
    delete obj[head];
    return obj;
  }
  obj[head] = setIn(obj[head], rest, value);
  return obj;
}

export function deepEqual(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  if (Array.isArray(a) && Array.isArray(b)) {
    return a.length === b.length && a.every((v, i) => deepEqual(v, b[i]));
  }
  if (isRecord(a) && isRecord(b)) {
    const ka = Object.keys(a);
    const kb = Object.keys(b);
    return ka.length === kb.length && ka.every((k) => Object.prototype.hasOwnProperty.call(b, k) && deepEqual(a[k], b[k]));
  }
  return false;
}

export interface DiffEntry {
  path: string;
  before: unknown;
  after: unknown;
}

/** Changed leaves between two trees, as dotted paths (lists by index); `undefined` = absent. */
export function diffTrees(before: unknown, after: unknown, path: PathSeg[] = []): DiffEntry[] {
  if (deepEqual(before, after)) return [];
  if (isRecord(before) && isRecord(after)) {
    const keys = [...Object.keys(before), ...Object.keys(after).filter((k) => !(k in before))];
    return keys.sort().flatMap((k) => diffTrees(before[k], after[k], [...path, k]));
  }
  if (Array.isArray(before) && Array.isArray(after)) {
    const n = Math.max(before.length, after.length);
    return Array.from({ length: n }, (_, i) => diffTrees(before[i], after[i], [...path, i])).flat();
  }
  return [{ path: toDotted(path), before, after }];
}

export { isRecord };
