// Video player of the deck: plays the original as soon as the browser can
// (or the background-encoded preview MP4 once it is ready), and loops the
// chosen segment.
import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import type { VideoRecord } from "../api/video";
import { Icon } from "../ui/icons";

export interface PlayerHandle {
  seek(seconds: number): void;
}

export interface PlayerProps {
  video: VideoRecord;
  /** Segment to loop over, in seconds. */
  loopRange: { start: number; end: number } | null;
  onTime?: (seconds: number) => void;
}

type Source = "original" | "preview";

export const Player = forwardRef<PlayerHandle, PlayerProps>(function Player({ video, loopRange, onTime }, ref) {
  const el = useRef<HTMLVideoElement>(null);
  const previewReady = video.preview_status === "ready";
  const [source, setSource] = useState<Source>(previewReady ? "preview" : "original");
  const [failed, setFailed] = useState<Source | null>(null);
  const range = useRef(loopRange);
  range.current = loopRange;

  // A new video starts from its best source.
  useEffect(() => {
    setSource(video.preview_status === "ready" ? "preview" : "original");
    setFailed(null);
    // only on a different video
  }, [video.id]);

  // The original did not play (VFR .mkv, HEVC...): switch once the preview exists.
  useEffect(() => {
    if (failed === "original" && previewReady) {
      setSource("preview");
      setFailed(null);
    }
  }, [failed, previewReady]);

  useImperativeHandle(ref, () => ({
    seek(seconds: number) {
      const v = el.current;
      if (v && Number.isFinite(seconds)) {
        try {
          v.currentTime = seconds;
        } catch {
          // not seekable yet
        }
      }
    },
  }));

  const src = source === "preview" ? video.urls.preview : video.urls.file;
  const waitingForPreview = failed === "original" && video.preview_status === "pending";
  const unplayable = failed !== null && !waitingForPreview;

  return (
    <div className="player">
      <video
        ref={el}
        key={`${video.id}:${source}`}
        className="player__video"
        src={src}
        poster={video.urls.thumb}
        autoPlay
        muted
        playsInline
        controls
        preload="auto"
        aria-label={`Video ${video.name}`}
        onError={() => setFailed(source)}
        onLoadedData={() => {
          const r = range.current;
          if (r && el.current && el.current.currentTime < r.start) el.current.currentTime = r.start;
        }}
        onTimeUpdate={(e) => {
          const v = e.currentTarget;
          const r = range.current;
          if (r && !v.paused && (v.currentTime >= r.end || v.currentTime < r.start - 0.25)) v.currentTime = r.start;
          onTime?.(v.currentTime);
        }}
        onSeeked={(e) => onTime?.(e.currentTarget.currentTime)}
      />
      {waitingForPreview ? (
        <div className="player__overlay" role="status">
          <span className="spinner" aria-hidden="true" />
          <p>This format does not play in the browser. Preparing a preview…</p>
          <p className="muted">Processing works on the original right away.</p>
        </div>
      ) : null}
      {unplayable ? (
        <div className="player__overlay" role="status">
          <Icon name="alert" size={22} />
          <p>The browser cannot play this video{video.preview_status === "failed" ? " and the preview could not be made" : ""}.</p>
          <p className="muted">It can still be processed.</p>
        </div>
      ) : null}
      {source === "preview" ? <span className="player__badge" title="Browser preview (re-encoded); processing uses the original">Preview</span> : null}
    </div>
  );
});
