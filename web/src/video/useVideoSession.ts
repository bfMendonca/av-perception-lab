// The video side of the studio: the library, the current video, an upload in
// progress and the chosen segment. Lives at the app level so a drop anywhere
// on the page, the top bar and the video deck share it.
import { useCallback, useEffect, useRef, useState } from "react";
import type { VideoApi, VideoRecord } from "../api/video";
import { MIN_SEGMENT_S } from "./segment";

const CURRENT_KEY = "ddd.video.current";
/** Default segment length: the ~3 s preview the spec asks for. */
export const DEFAULT_SEGMENT_S = 3;
const PREVIEW_POLL_MS = 2000;

export interface UploadState {
  name: string;
  loaded: number;
  total: number;
  /** `processing`: the bytes are in, the server probes / converts the video. */
  phase: "uploading" | "processing";
}

export interface Segment {
  start: number;
  length: number;
}

export interface SessionOptions {
  /** Remember the current video across reloads (localStorage). */
  remember?: boolean;
  initial?: VideoRecord | null;
  onError?: (text: string) => void;
}

function readStored(): string | null {
  try {
    return window.localStorage.getItem(CURRENT_KEY);
  } catch {
    return null;
  }
}

function writeStored(id: string | null) {
  try {
    if (id) window.localStorage.setItem(CURRENT_KEY, id);
    else window.localStorage.removeItem(CURRENT_KEY);
  } catch {
    // storage unavailable: the choice just does not survive a reload
  }
}

export function defaultSegment(record: VideoRecord | null): Segment {
  const total = record?.duration_s ?? 0;
  return { start: 0, length: total > 0 ? Math.min(DEFAULT_SEGMENT_S, total) : DEFAULT_SEGMENT_S };
}

/** Keep a segment inside [0, total] and at least MIN_SEGMENT_S long. */
export function clampSegment(seg: Segment, total: number): Segment {
  if (!(total > 0)) return seg;
  const length = Math.min(Math.max(seg.length, Math.min(MIN_SEGMENT_S, total)), total);
  const start = Math.min(Math.max(seg.start, 0), total - length);
  return { start, length };
}

export function useVideoSession(api: VideoApi, options: SessionOptions = {}) {
  const { remember = false, initial = null } = options;
  const onError = useRef(options.onError);
  onError.current = options.onError;
  const [videos, setVideos] = useState<VideoRecord[]>([]);
  const [current, setCurrent] = useState<VideoRecord | null>(initial);
  const [segment, setSegmentState] = useState<Segment>(() => defaultSegment(initial));
  const [stride, setStride] = useState(1);
  const [upload, setUpload] = useState<UploadState | null>(null);
  const [roots, setRoots] = useState<string[]>([]);
  const [cwd, setCwd] = useState("");
  const [error, setError] = useState<string | null>(null);
  const abort = useRef<AbortController | null>(null);

  const fail = useCallback((text: string) => {
    setError(text);
    onError.current?.(text);
  }, []);

  const select = useCallback(
    (record: VideoRecord | null) => {
      setCurrent(record);
      setSegmentState(defaultSegment(record));
      setError(null);
      if (remember) writeStored(record?.id ?? null);
      if (record) setVideos((list) => [record, ...list.filter((v) => v.id !== record.id)]);
    },
    [remember],
  );

  const refreshLibrary = useCallback(async () => {
    try {
      const list = await api.listVideos();
      setVideos(list);
      return list;
    } catch {
      return [];
    }
  }, [api]);

  useEffect(() => {
    let live = true;
    void refreshLibrary().then((list) => {
      if (!live || !remember || initial) return;
      const stored = readStored();
      const found = stored ? list.find((v) => v.id === stored) : undefined;
      if (found) {
        setCurrent(found);
        setSegmentState(defaultSegment(found));
      }
    });
    api
      .importRoots()
      .then((r) => {
        if (!live) return;
        setRoots(r.roots);
        setCwd(r.cwd);
      })
      .catch(() => undefined);
    return () => {
      live = false;
    };
    // `initial` and `remember` are construction options: read once.
  }, [api, refreshLibrary]);

  // The preview MP4 is encoded in the background: poll until it settles.
  const pendingId = current?.preview_status === "pending" ? current.id : null;
  useEffect(() => {
    if (!pendingId) return;
    const timer = setInterval(() => {
      api
        .getVideo(pendingId)
        .then((record) => {
          if (record.preview_status === "pending") return;
          setCurrent((cur) => (cur?.id === record.id ? record : cur));
          setVideos((list) => list.map((v) => (v.id === record.id ? record : v)));
        })
        .catch(() => undefined);
    }, PREVIEW_POLL_MS);
    return () => clearInterval(timer);
  }, [api, pendingId]);

  const uploadFile = useCallback(
    async (file: File, meta?: File) => {
      abort.current?.abort();
      const controller = new AbortController();
      abort.current = controller;
      setError(null);
      setUpload({ name: file.name, loaded: 0, total: file.size, phase: "uploading" });
      try {
        const record = await api.upload(file, {
          meta,
          signal: controller.signal,
          onProgress: (loaded, total) =>
            setUpload({ name: file.name, loaded, total, phase: loaded >= total && total > 0 ? "processing" : "uploading" }),
        });
        select(record);
        return record;
      } catch (exc) {
        if (!(exc instanceof DOMException && exc.name === "AbortError")) {
          fail(`Could not add ${file.name}: ${exc instanceof Error ? exc.message : String(exc)}`);
        }
        return null;
      } finally {
        if (abort.current === controller) {
          abort.current = null;
          setUpload(null);
        }
      }
    },
    [api, fail, select],
  );

  const importPath = useCallback(
    async (path: string) => {
      const name = path.split(/[\\/]/).pop() || path;
      setError(null);
      setUpload({ name, loaded: 0, total: 0, phase: "processing" });
      try {
        const record = await api.importPath(path.trim());
        select(record);
        return record;
      } catch (exc) {
        fail(`Could not import ${name}: ${exc instanceof Error ? exc.message : String(exc)}`);
        return null;
      } finally {
        setUpload(null);
      }
    },
    [api, fail, select],
  );

  const setSegment = useCallback(
    (seg: Segment) => setSegmentState(clampSegment(seg, current?.duration_s ?? 0)),
    [current],
  );

  return {
    videos,
    current,
    segment,
    stride,
    upload,
    roots,
    cwd,
    error,
    select,
    setSegment,
    setStride: (n: number) => setStride(Math.max(1, Math.round(n) || 1)),
    uploadFile,
    importPath,
    cancelUpload: () => abort.current?.abort(),
    dismissError: () => setError(null),
    refreshLibrary,
  };
}

export type VideoSession = ReturnType<typeof useVideoSession>;
