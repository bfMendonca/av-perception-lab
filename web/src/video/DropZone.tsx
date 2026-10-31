// Getting a video in: drop it anywhere on the page, browse for it, paste a
// path on the server, or pick one already in the library.
import { useRef, useState } from "react";
import type { VideoRecord } from "../api/video";
import { Icon } from "../ui/icons";

const VIDEO_EXT = /\.(mp4|m4v|mov|mkv|webm|avi|mts|m2ts|ts|h264|264|hevc|h265)$/i;
const YAML_EXT = /\.ya?ml$/i;
const META_EXT = /\.meta\.json$/i;

export interface DroppedFiles {
  yaml?: File;
  video?: File;
  /** The `.meta.json` dropped with a raw `.h264` (sensor timestamps). */
  meta?: File;
}

/** Sort a drop into a pipeline YAML, a video and its companion `.meta.json`. */
export function classifyFiles(files: Iterable<File>): DroppedFiles {
  const out: DroppedFiles = {};
  for (const file of files) {
    if (META_EXT.test(file.name)) out.meta ??= file;
    else if (YAML_EXT.test(file.name)) out.yaml ??= file;
    else if (file.type.startsWith("video/") || VIDEO_EXT.test(file.name)) out.video ??= file;
  }
  return out;
}

export function isRawH264(name: string): boolean {
  return /\.(h264|264)$/i.test(name);
}

export function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let v = n / 1024;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v < 10 ? v.toFixed(1) : Math.round(v)} ${units[i]}`;
}

/** Collapse `.` and `..` of a POSIX path (no file system access). */
function normalize(path: string): string {
  const out: string[] = [];
  for (const part of path.split("/")) {
    if (!part || part === ".") continue;
    if (part === "..") out.pop();
    else out.push(part);
  }
  return `/${out.join("/")}`;
}

/**
 * Whether the server would accept importing `path`: relative paths start at
 * `cwd`, and the result must sit inside one of `roots`. Symlinks are only
 * known to the server, which checks again.
 */
export function importable(path: string, roots: string[], cwd: string): boolean {
  if (!path || !roots.length) return false;
  if (!path.startsWith("/") && !cwd) return false;
  const full = normalize(path.startsWith("/") ? path : `${cwd}/${path}`);
  return roots.some((r) => {
    const root = normalize(r);
    return full === root || full.startsWith(root === "/" ? "/" : `${root}/`);
  });
}

export interface DropZoneProps {
  videos: VideoRecord[];
  roots: string[];
  /** The server's working directory (relative paths start there). */
  cwd: string;
  /** A path from the open pipeline (source.video) offered for import. */
  suggestedPath?: string | null;
  busy: boolean;
  onFiles: (files: DroppedFiles) => void;
  onImport: (path: string) => void;
  onSelect: (video: VideoRecord) => void;
}

export function DropZone({ videos, roots, cwd, suggestedPath, busy, onFiles, onImport, onSelect }: DropZoneProps) {
  const input = useRef<HTMLInputElement>(null);
  const [path, setPath] = useState("");
  return (
    <div className="intake">
      <div className="intake__drop">
        <div className="intake__art" aria-hidden="true">
          <Icon name="film" size={26} />
        </div>
        <div className="intake__text">
          <h2>Drop a video anywhere</h2>
          <p>
            MP4, MOV or MKV play right away. A Raspberry Pi <code>.h264</code> goes in together with its <code>.meta.json</code> and is converted with the sensor timestamps.
          </p>
        </div>
        <input
          ref={input}
          type="file"
          multiple
          hidden
          accept="video/*,.mkv,.h264,.264,.meta.json,.json"
          data-testid="video-input"
          onChange={(e) => {
            const files = classifyFiles(Array.from(e.target.files ?? []));
            e.target.value = "";
            if (files.video) onFiles(files);
          }}
        />
        <button type="button" className="btn btn--primary" disabled={busy} onClick={() => input.current?.click()}>
          <Icon name="upload" size={15} /> Choose video
        </button>
      </div>
      <form
        className="intake__path"
        onSubmit={(e) => {
          e.preventDefault();
          if (path.trim()) onImport(path.trim());
        }}
      >
        <label htmlFor="server-path">Or a file on the server</label>
        <div className="intake__row">
          <span className="text text--path">
            <span className="text__icon" aria-hidden="true">
              <Icon name="folder" size={14} />
            </span>
            <input
              id="server-path"
              className="input input--mono"
              placeholder={roots[0] ? `${roots[0]}/…/video.mkv` : "/path/to/video.mp4"}
              value={path}
              spellCheck={false}
              autoComplete="off"
              onChange={(e) => setPath(e.target.value)}
            />
          </span>
          <button type="submit" className="btn" disabled={busy || !path.trim()}>
            Import
          </button>
        </div>
        <p className="intake__hint">
          {roots.length ? (
            <>
              Allowed folders: {roots.map((r, i) => (
                <code key={r}>
                  {r}
                  {i < roots.length - 1 ? ", " : ""}
                </code>
              ))}
              . Imports are referenced, not copied.
            </>
          ) : (
            "Server imports are off (start the server with --data-root)."
          )}
        </p>
        {suggestedPath && importable(suggestedPath, roots, cwd) ? (
          <button type="button" className="btn btn--sm btn--ghost intake__suggest" disabled={busy} onClick={() => onImport(suggestedPath)}>
            <Icon name="file" size={13} /> Use the pipeline's video <code>{suggestedPath}</code>
          </button>
        ) : null}
      </form>
      {videos.length ? (
        <div className="intake__library">
          <h3>Library</h3>
          <ul>
            {videos.slice(0, 8).map((v) => (
              <li key={v.id}>
                <button type="button" className="libitem" onClick={() => onSelect(v)} title={v.original}>
                  <img src={v.urls.thumb} alt="" loading="lazy" />
                  {v.missing ? <span className="libitem__missing">File missing</span> : null}
                  <span className="libitem__name">{v.name}</span>
                  <span className="libitem__meta">
                    {v.width}×{v.height} · {Math.round(v.duration_s)} s
                  </span>
                </button>
              </li>
            ))}
          </ul>
        </div>
      ) : null}
    </div>
  );
}
