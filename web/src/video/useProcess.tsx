// The Process action: the editor's pipeline plus the current video and
// segment, after a free-GPU-memory check that asks before queueing when
// another job (a batch run, say) holds the device.
import { useCallback, useState } from "react";
import { RunRejected, type VideoApi } from "../api/video";
import type { useEditor } from "../editor/yamlState";
import { Dialog } from "../ui/Dialog";
import { buildRunConfig, estimateSeconds, expectedFrames, formatDuration, formatTime, segmentToSource, type SourceSegment } from "./segment";
import type { Jobs } from "./useJobs";
import type { VideoSession } from "./useVideoSession";

/** Below this much free device memory (MiB) Process asks for confirmation. */
export const GPU_MIN_FREE_MIB = 4000;

export interface ProcessPlan {
  segment: SourceSegment;
  frames: number;
  seconds: number;
  label: string;
}

type Editor = ReturnType<typeof useEditor>;

export function useProcess(api: VideoApi, session: VideoSession, jobs: Jobs, editor: Editor) {
  const [confirm, setConfirm] = useState<{ entire: boolean; free: number } | null>(null);
  const [busy, setBusy] = useState(false);
  const video = session.current;

  const blocked: string | null = !video
    ? "Load a video first."
    : !editor.loaded
      ? "Open or start a pipeline first."
      : editor.validation === "invalid"
        ? "Fix the pipeline's problems first."
        : session.upload
          ? "Wait for the upload to finish."
          : null;

  const plan = useCallback(
    (entire: boolean): ProcessPlan | null => {
      if (!video) return null;
      const segment = entire
        ? segmentToSource(0, 0, session.stride, video.duration_s)
        : segmentToSource(session.segment.start, session.segment.length, session.stride, video.duration_s);
      const length = segment.duration > 0 ? segment.duration : video.duration_s - segment.start;
      const frames = expectedFrames(length, video.fps, segment.stride);
      const range = entire ? "entire video" : `${formatTime(segment.start)}–${formatTime(segment.start + length)}`;
      return {
        segment,
        frames,
        seconds: estimateSeconds(frames),
        label: `${video.name} · ${range}${segment.stride > 1 ? ` · stride ${segment.stride}` : ""}`,
      };
    },
    [session.segment, session.stride, video],
  );

  const submit = useCallback(
    async (entire: boolean) => {
      const p = plan(entire);
      if (!p || !video) return;
      setBusy(true);
      try {
        const config = buildRunConfig(editor.toConfig(), video.original, p.segment);
        await jobs.submit(config, video.id, { label: p.label, expected: p.frames });
      } catch (exc) {
        const text =
          exc instanceof RunRejected
            ? `The server rejected the run (${exc.errors.length} problem${exc.errors.length === 1 ? "" : "s"}): ${exc.message}`
            : `Could not start the run: ${exc instanceof Error ? exc.message : String(exc)}`;
        editor.notify({ tone: "error", text });
      } finally {
        setBusy(false);
      }
    },
    [editor, jobs, plan, video],
  );

  const process = useCallback(
    async (entire = false) => {
      if (blocked || busy) return;
      setBusy(true);
      let free: number | null = null;
      try {
        free = await api.gpuFreeMiB();
      } catch {
        free = null; // unknown: do not block
      }
      if (free !== null && free < GPU_MIN_FREE_MIB) {
        setBusy(false);
        setConfirm({ entire, free });
        return;
      }
      await submit(entire);
    },
    [api, blocked, busy, submit],
  );

  const pendingPlan = confirm ? plan(confirm.entire) : null;
  const dialog = (
    <Dialog
      open={confirm !== null}
      title="The GPU is busy"
      onClose={() => setConfirm(null)}
      actions={
        <>
          <button type="button" className="btn" onClick={() => setConfirm(null)}>
            Cancel
          </button>
          <button
            type="button"
            className="btn btn--primary"
            autoFocus
            onClick={() => {
              const entire = confirm?.entire ?? false;
              setConfirm(null);
              void submit(entire);
            }}
          >
            Process anyway
          </button>
        </>
      }
    >
      <p>
        Only <strong>{confirm?.free.toLocaleString("en-US")} MiB</strong> of GPU memory is free; a run needs about {GPU_MIN_FREE_MIB.toLocaleString("en-US")} MiB. Another process (a batch run, for example) is probably using the device, and this run may fail with an out-of-memory error.
      </p>
      {pendingPlan ? (
        <p className="muted">
          {pendingPlan.label} · {pendingPlan.frames} frames, {formatDuration(pendingPlan.seconds)} when the GPU is free.
        </p>
      ) : null}
    </Dialog>
  );

  return { process, plan, blocked, busy, dialog };
}

export type Process = ReturnType<typeof useProcess>;
