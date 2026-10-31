// Distance of one track over time: run B (the selected run) drawn over run A
// (the comparison run, dashed), B's raw measurement in grey under its
// smoothed line. One <path> per series (the server already decimates to
// ~1500 points), so a 5-minute clip stays responsive. Clicking (or Enter on
// the keyboard cursor) seeks the annotated video to that point.
import { useEffect, useMemo, useRef, useState, type KeyboardEvent, type MouseEvent, type RefObject } from "react";
import type { Series } from "../api/runs";
import { formatNumber, formatSeconds } from "./format";

export const PAD = { left: 48, right: 16, top: 14, bottom: 30 };
const HEIGHT = 260;
const DEFAULT_WIDTH = 800;

export type Scale = ((v: number) => number) & { invert: (p: number) => number };

/** Linear map of `domain` onto `range` (a zero-width domain maps to the middle). */
export function makeScale([d0, d1]: [number, number], [r0, r1]: [number, number]): Scale {
  const span = d1 - d0;
  const f = ((v: number) => (span ? r0 + ((v - d0) / span) * (r1 - r0) : (r0 + r1) / 2)) as Scale;
  f.invert = (p: number) => (r1 === r0 ? d0 : d0 + ((p - r0) / (r1 - r0)) * span);
  return f;
}

/** Index of the value of a sorted array closest to `x` (-1 when empty). */
export function nearestIndex(sorted: number[], x: number): number {
  if (!sorted.length) return -1;
  let lo = 0;
  let hi = sorted.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (sorted[mid] < x) lo = mid;
    else hi = mid;
  }
  return Math.abs(sorted[lo] - x) <= Math.abs(sorted[hi] - x) ? lo : hi;
}

export function niceTicks(min: number, max: number, count = 5): number[] {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [];
  if (min === max) return [min];
  const raw = (max - min) / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw;
  const ticks: number[] = [];
  for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-9; v += step) ticks.push(Number(v.toFixed(10)));
  return ticks;
}

function linePath(t: number[], values: (number | null)[], x: Scale, y: Scale): string {
  let d = "";
  let pen = false;
  for (let i = 0; i < t.length; i++) {
    const v = values[i];
    if (v === null || v === undefined || !Number.isFinite(v)) {
      pen = false;
      continue;
    }
    d += `${pen ? "L" : "M"}${x(t[i]).toFixed(1)} ${y(v).toFixed(1)}`;
    pen = true;
  }
  return d;
}

function extent(lists: (number | null)[][]): [number, number] {
  let lo = Infinity;
  let hi = -Infinity;
  for (const list of lists)
    for (const v of list)
      if (v !== null && Number.isFinite(v)) {
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
  return [lo, hi];
}

function useWidth(): [RefObject<HTMLDivElement>, number] {
  const ref = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(DEFAULT_WIDTH);
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(([entry]) => {
      const w = Math.round(entry.contentRect.width);
      if (w > 0) setWidth(w);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, width];
}

export interface DistanceChartProps {
  /** The selected run (seeks its video). */
  b: Series | null;
  /** The comparison run, drawn dashed underneath. */
  a?: Series | null;
  labelB: string;
  labelA?: string;
  /** Seek the annotated video of B to this time (s). */
  onSeek: (videoT: number) => void;
  /** Current time of B's video, drawn as a playhead. */
  playhead?: number | null;
}

function mainValues(s: Series): (number | null)[] {
  return s.smoothed ?? s.raw;
}

export function DistanceChart({ b, a, labelB, labelA, onSeek, playhead }: DistanceChartProps) {
  const [box, width] = useWidth();
  const svgRef = useRef<SVGSVGElement>(null);
  const [cursor, setCursor] = useState<number | null>(null);
  const plotW = Math.max(40, width - PAD.left - PAD.right);
  const plotH = HEIGHT - PAD.top - PAD.bottom;

  const geom = useMemo(() => {
    const series = [b, a].filter((s): s is Series => Boolean(s && s.t.length));
    if (!series.length) return null;
    const [t0, t1] = extent(series.map((s) => s.t));
    const [v0, v1] = extent(series.flatMap((s) => [s.raw, ...(s.smoothed ? [s.smoothed] : [])]));
    if (!Number.isFinite(t0) || !Number.isFinite(v0)) return null;
    const padV = (v1 - v0) * 0.08 || Math.max(1, Math.abs(v0) * 0.05);
    const yTicks = niceTicks(Math.max(0, v0 - padV), v1 + padV, 4);
    const yDomain: [number, number] = [Math.min(yTicks[0] ?? v0, v0), Math.max(yTicks[yTicks.length - 1] ?? v1, v1)];
    const x = makeScale([t0, t1], [PAD.left, PAD.left + plotW]);
    const y = makeScale(yDomain, [PAD.top + plotH, PAD.top]);
    return { x, y, xTicks: niceTicks(t0, t1, Math.max(2, Math.round(plotW / 110))), yTicks: yTicks.filter((v) => v >= yDomain[0] && v <= yDomain[1]) };
  }, [a, b, plotW, plotH]);

  if (!b || !geom) {
    return (
      <div className="chart chart--empty" ref={box}>
        <p className="muted">{b && !b.found ? "This track is not in the run." : "No distances to plot for this run."}</p>
      </div>
    );
  }
  const { x, y } = geom;
  const bMain = mainValues(b);
  const aMain = a && a.t.length ? mainValues(a) : null;
  const playIdx = playhead !== null && playhead !== undefined ? nearestIndex(b.video_t.map((v) => v ?? -Infinity), playhead) : -1;

  const indexAt = (clientX: number): number => {
    const rect = svgRef.current?.getBoundingClientRect();
    const left = rect ? rect.left : 0;
    const scaleX = rect && rect.width ? width / rect.width : 1;
    return nearestIndex(b.t, x.invert((clientX - left) * scaleX));
  };
  const seek = (i: number) => {
    const v = b.video_t[i];
    if (i >= 0 && v !== null && v !== undefined) onSeek(Math.max(0, v));
  };
  const onKey = (e: KeyboardEvent) => {
    const n = b.t.length;
    const step = e.shiftKey ? Math.max(1, Math.round(n / 20)) : 1;
    if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
      e.preventDefault();
      const from = cursor ?? (e.key === "ArrowRight" ? -1 : n);
      setCursor(Math.min(n - 1, Math.max(0, from + (e.key === "ArrowRight" ? step : -step))));
    } else if (e.key === "Enter" && cursor !== null) {
      e.preventDefault();
      seek(cursor);
    } else if (e.key === "Escape") setCursor(null);
  };

  const hover = cursor !== null && cursor >= 0 ? cursor : null;
  const aIdx = hover !== null && a && a.t.length ? nearestIndex(a.t, b.t[hover]) : -1;
  const aNear = aIdx >= 0 && a && Math.abs(a.t[aIdx] - b.t[hover as number]) <= 0.5 ? aIdx : -1;
  const tipLeft = hover !== null ? x(b.t[hover]) : 0;

  return (
    <div className="chart" ref={box}>
      <svg
        ref={svgRef}
        className="chart__svg"
        width={width}
        height={HEIGHT}
        viewBox={`0 0 ${width} ${HEIGHT}`}
        role="img"
        aria-label={`Distance over time of ${labelB}${aMain ? ` over ${labelA ?? "the comparison run"}` : ""}. Click a point to seek the video.`}
      >
        <g className="chart__grid">
          {geom.yTicks.map((v) => (
            <g key={`y${v}`}>
              <line x1={PAD.left} x2={PAD.left + plotW} y1={y(v)} y2={y(v)} />
              <text x={PAD.left - 8} y={y(v)} dy="0.32em" textAnchor="end">
                {formatNumber(v)}
              </text>
            </g>
          ))}
          {geom.xTicks.map((v) => (
            <text key={`x${v}`} x={x(v)} y={HEIGHT - 10} textAnchor="middle">
              {formatSeconds(v)}
            </text>
          ))}
          <text x={PAD.left - 8} y={PAD.top - 4} textAnchor="end" className="chart__unit">
            m
          </text>
        </g>
        {aMain && a ? <path className="chart__line chart__line--a" d={linePath(a.t, aMain, x, y)} /> : null}
        {b.smoothed ? <path className="chart__line chart__line--raw" d={linePath(b.t, b.raw, x, y)} /> : null}
        <path className="chart__line chart__line--b" d={linePath(b.t, bMain, x, y)} />
        {playIdx >= 0 ? <line className="chart__playhead" x1={x(b.t[playIdx])} x2={x(b.t[playIdx])} y1={PAD.top} y2={PAD.top + plotH} /> : null}
        {hover !== null ? (
          <g className="chart__cursor">
            <line x1={tipLeft} x2={tipLeft} y1={PAD.top} y2={PAD.top + plotH} />
            {bMain[hover] !== null ? <circle cx={tipLeft} cy={y(bMain[hover] as number)} r={4} /> : null}
          </g>
        ) : null}
        <rect
          data-testid="chart-plot"
          className="chart__hit"
          x={PAD.left}
          y={PAD.top}
          width={plotW}
          height={plotH}
          tabIndex={0}
          aria-label="Chart cursor: arrow keys move, Enter seeks the video"
          onMouseMove={(e: MouseEvent) => setCursor(indexAt(e.clientX))}
          onMouseLeave={() => setCursor(null)}
          onClick={(e: MouseEvent) => seek(indexAt(e.clientX))}
          onKeyDown={onKey}
          onBlur={() => setCursor(null)}
        />
      </svg>
      {hover !== null ? (
        <div className="chart__tip" style={{ left: Math.min(Math.max(tipLeft, 90), width - 90) }} role="status">
          <strong>{formatSeconds(b.t[hover])}</strong>
          <span>
            <i className="key key--b" /> {bMain[hover] !== null ? `${formatNumber(bMain[hover] as number)} m` : "n/a"} <em>{labelB}</em>
          </span>
          {b.smoothed ? (
            <span>
              <i className="key key--raw" /> {b.raw[hover] !== null ? `${formatNumber(b.raw[hover] as number)} m` : "n/a"} <em>raw</em>
            </span>
          ) : null}
          {aMain && aNear >= 0 && a ? (
            <span>
              <i className="key key--a" /> {aMain[aNear] !== null ? `${formatNumber(aMain[aNear] as number)} m` : "n/a"} <em>{labelA}</em>
            </span>
          ) : null}
        </div>
      ) : null}
      <div className="chart__legend">
        <span>
          <i className="key key--b" /> {labelB}
          {b.smoothed ? " · smoothed" : ""}
        </span>
        {b.smoothed ? (
          <span>
            <i className="key key--raw" /> {labelB} · raw
          </span>
        ) : null}
        {aMain ? (
          <span>
            <i className="key key--a" /> {labelA}
            {a?.smoothed ? " · smoothed" : ""}
          </span>
        ) : null}
        <span className="chart__hint">Click a point to seek the video{b.downsampled ? ` · ${b.t.length} of ${b.points} points shown` : ""}</span>
      </div>
    </div>
  );
}
