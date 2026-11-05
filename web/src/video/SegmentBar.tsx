// Segment bar: the whole video as a track, the processed segment between two
// handles (drag a handle, drag the segment, or use the keyboard), the play
// head, and the numeric start / length / stride fields with a live estimate.
import { useRef, useState, type KeyboardEvent, type PointerEvent } from "react";
import { Icon } from "../ui/icons";
import { estimateSeconds, expectedFrames, formatDuration, formatTime, MIN_SEGMENT_S, PROCESSING_FPS } from "./segment";
import type { Segment } from "./useVideoSession";

export interface SegmentBarProps {
  duration: number;
  fps: number;
  /** Frames in the input when it has no rate (an MCAP topic with one message). */
  count?: number;
  segment: Segment;
  stride: number;
  playhead: number;
  onChange: (segment: Segment) => void;
  onStride: (stride: number) => void;
  /** Move the play head (click on the track, a handle moved). */
  onSeek: (seconds: number) => void;
}

type Drag = { kind: "start" | "end" | "move"; offset: number } | null;

const round = (s: number) => Math.round(s * 100) / 100;

export function SegmentBar({ duration, fps, count = 0, segment, stride, playhead, onChange, onStride, onSeek }: SegmentBarProps) {
  const track = useRef<HTMLDivElement>(null);
  const [drag, setDrag] = useState<Drag>(null);
  const total = duration > 0 ? duration : 1;
  const end = segment.start + segment.length;
  const pct = (t: number) => `${(Math.min(Math.max(t / total, 0), 1) * 100).toFixed(3)}%`;
  const step = fps > 0 ? Math.max(1 / fps, 0.04) : 0.1;
  const minLen = Math.min(MIN_SEGMENT_S, total);

  const timeAt = (clientX: number) => {
    const rect = track.current?.getBoundingClientRect();
    if (!rect || rect.width <= 0) return 0;
    return Math.min(Math.max((clientX - rect.left) / rect.width, 0), 1) * total;
  };

  const setStart = (t: number) => {
    const start = Math.min(Math.max(t, 0), end - minLen);
    onChange({ start: round(start), length: round(end - start) });
    onSeek(start);
  };
  /** Move the whole segment to start at `t` (the length is kept). */
  const moveTo = (t: number) => {
    const start = Math.min(Math.max(t, 0), Math.max(total - segment.length, 0));
    onChange({ start: round(start), length: segment.length });
    onSeek(start);
  };
  const setEnd = (t: number) => {
    const e = Math.max(Math.min(t, total), segment.start + minLen);
    onChange({ start: segment.start, length: round(e - segment.start) });
    onSeek(Math.max(segment.start, e - 0.5));
  };

  const begin = (kind: "start" | "end" | "move") => (e: PointerEvent<HTMLElement>) => {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.setPointerCapture?.(e.pointerId);
    setDrag({ kind, offset: kind === "move" ? timeAt(e.clientX) - segment.start : 0 });
  };
  const move = (e: PointerEvent<HTMLElement>) => {
    if (!drag) return;
    const t = timeAt(e.clientX);
    if (drag.kind === "start") setStart(t);
    else if (drag.kind === "end") setEnd(t);
    else moveTo(t - drag.offset);
  };
  const stop = () => setDrag(null);

  const keys = (which: "start" | "end") => (e: KeyboardEvent<HTMLElement>) => {
    const big = e.shiftKey ? 10 : 1;
    const value = which === "start" ? segment.start : end;
    let next: number | null = null;
    if (e.key === "ArrowLeft" || e.key === "ArrowDown") next = value - step * big;
    else if (e.key === "ArrowRight" || e.key === "ArrowUp") next = value + step * big;
    else if (e.key === "PageDown") next = value - 1;
    else if (e.key === "PageUp") next = value + 1;
    else if (e.key === "Home") next = 0;
    else if (e.key === "End") next = total;
    if (next === null) return;
    e.preventDefault();
    if (which === "start") setStart(next);
    else setEnd(next);
  };

  const frames = expectedFrames(segment.length, fps, stride, count);
  const estimate = estimateSeconds(frames);

  return (
    <div className="segment">
      <div
        ref={track}
        className={`segment__track${drag ? " segment__track--drag" : ""}`}
        onPointerDown={(e) => {
          if (e.target === e.currentTarget || (e.target as HTMLElement).dataset.part === "rail") onSeek(timeAt(e.clientX));
        }}
        onPointerMove={move}
        onPointerUp={stop}
        onPointerCancel={stop}
      >
        <div className="segment__rail" data-part="rail" />
        <div className="segment__range" style={{ left: pct(segment.start), width: `calc(${pct(end)} - ${pct(segment.start)})` }} onPointerDown={begin("move")} title="Drag to move the segment" />
        <div className="segment__playhead" style={{ left: pct(playhead) }} aria-hidden="true" />
        <div
          className="segment__handle segment__handle--start"
          style={{ left: pct(segment.start) }}
          role="slider"
          tabIndex={0}
          aria-label="Segment start"
          aria-valuemin={0}
          aria-valuemax={round(total)}
          aria-valuenow={round(segment.start)}
          aria-valuetext={formatTime(segment.start)}
          onPointerDown={begin("start")}
          onKeyDown={keys("start")}
        />
        <div
          className="segment__handle segment__handle--end"
          style={{ left: pct(end) }}
          role="slider"
          tabIndex={0}
          aria-label="Segment end"
          aria-valuemin={0}
          aria-valuemax={round(total)}
          aria-valuenow={round(end)}
          aria-valuetext={formatTime(end)}
          onPointerDown={begin("end")}
          onKeyDown={keys("end")}
        />
      </div>
      <div className="segment__scale" aria-hidden="true">
        <span>0:00.0</span>
        <span>{formatTime(total / 2)}</span>
        <span>{formatTime(total)}</span>
      </div>
      <div className="segment__fields">
        <NumberField label="Start" unit="s" value={round(segment.start)} min={0} max={round(total - minLen)} step={0.1} onCommit={moveTo} />
        <NumberField label="Length" unit="s" value={round(segment.length)} min={minLen} max={round(total)} step={0.1} onCommit={(v) => onChange({ start: segment.start, length: v })} />
        <NumberField label="Stride" value={stride} min={1} max={1000} step={1} onCommit={onStride} hint="Process every N-th frame" />
        <div className="segment__marks">
          <button type="button" className="btn btn--xs btn--ghost" title="Start the segment at the play head" onClick={() => (playhead < end - minLen ? setStart(playhead) : moveTo(playhead))}>
            <Icon name="mark-in" size={13} /> In
          </button>
          <button type="button" className="btn btn--xs btn--ghost" title="End the segment at the play head" onClick={() => setEnd(playhead)}>
            Out <Icon name="mark-out" size={13} />
          </button>
        </div>
      </div>
      <p className="segment__estimate" aria-live="polite" data-testid="segment-estimate">
        <span>
          {formatTime(segment.start)} – {formatTime(end)}
        </span>
        <span className="segment__dot" aria-hidden="true">·</span>
        <span>
          {frames} frame{frames === 1 ? "" : "s"}
        </span>
        <span className="segment__dot" aria-hidden="true">·</span>
        <strong>
          {formatDuration(estimate)} at {PROCESSING_FPS} fps
        </strong>
      </p>
    </div>
  );
}

function NumberField({ label, unit, value, min, max, step, onCommit, hint }: { label: string; unit?: string; value: number; min: number; max: number; step: number; onCommit: (v: number) => void; hint?: string }) {
  const [text, setText] = useState<string | null>(null);
  const commit = () => {
    if (text === null) return;
    const v = Number(text);
    if (text.trim() !== "" && Number.isFinite(v)) onCommit(Math.min(Math.max(v, min), max));
    setText(null);
  };
  return (
    <label className="segment__field" title={hint}>
      <span>{label}</span>
      <span className="segment__input">
        <input
          className="input input--num"
          type="number"
          inputMode="decimal"
          aria-label={label}
          value={text ?? String(value)}
          min={min}
          max={max}
          step={step}
          onChange={(e) => setText(e.target.value)}
          onBlur={commit}
          onKeyDown={(e) => {
            if (e.key === "Enter") commit();
            if (e.key === "Escape") setText(null);
          }}
        />
        {unit ? <em>{unit}</em> : null}
      </span>
    </label>
  );
}
