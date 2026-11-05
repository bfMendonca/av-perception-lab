// The image topics of an MCAP input: one radio per topic (name, resolution,
// rate, message count, CameraInfo badge); topics the reader cannot decode are
// disabled with the reason. Below, "use the CameraInfo" for the intrinsics.
import type { McapTopic } from "../api/types";
import { cameraInfoAvailability, formatRate, formatResolution } from "./mcapCamera";

export interface TopicPickerProps {
  topics: McapTopic[];
  /** The selected topic name (`source.topic`). */
  value: string | null;
  onChange: (topic: string) => void;
  /** `camera.calibration.from_source` is on. */
  useCameraInfo: boolean;
  onUseCameraInfo: (on: boolean) => void;
}

const shortSchema = (schema: string) => schema.split("/").pop() || schema;

function CameraInfoBadge({ topic }: { topic: McapTopic }) {
  if (topic.cameraInfoError) {
    return (
      <span className="chip chip--warn topic__badge" title={topic.cameraInfoError}>
        CameraInfo unusable
      </span>
    );
  }
  if (!topic.hasCameraInfo) return <span className="chip chip--muted topic__badge">no CameraInfo</span>;
  return (
    <span className="chip chip--ok topic__badge" title={topic.cameraInfoTopic ?? undefined}>
      {topic.hasDistortion ? "CameraInfo" : "CameraInfo · rectified"}
    </span>
  );
}

export function TopicPicker({ topics, value, onChange, useCameraInfo, onUseCameraInfo }: TopicPickerProps) {
  const selected = topics.find((t) => t.name === value) ?? null;
  const camera = cameraInfoAvailability(selected);
  return (
    <fieldset className="topics" aria-label="Image topics">
      <legend className="topics__legend">Image topic</legend>
      <ul className="topics__list">
        {topics.map((t) => (
          <li key={t.name}>
            <label className={`topic${t.name === value ? " topic--on" : ""}${t.supported ? "" : " topic--off"}`}>
              <input type="radio" name="mcap-topic" value={t.name} checked={t.name === value} disabled={!t.supported} onChange={() => onChange(t.name)} />
              <span className="topic__main">
                <span className="topic__name" title={t.name}>
                  {t.name}
                </span>
                <span className="topic__meta">
                  {formatResolution(t.width, t.height)} · {formatRate(t.rateHz, t.count)}
                  {t.rateHz > 0 ? ` · ${t.count} msgs` : ""} · {shortSchema(t.schema)}
                </span>
                {!t.supported && t.unsupportedReason ? <span className="topic__reason">{t.unsupportedReason}</span> : null}
              </span>
              <CameraInfoBadge topic={t} />
            </label>
          </li>
        ))}
      </ul>
      <label className={`topics__camera${camera.available ? "" : " topics__camera--off"}`}>
        <input type="checkbox" checked={useCameraInfo} disabled={!camera.available && !useCameraInfo} onChange={(e) => onUseCameraInfo(e.target.checked)} />
        <span>
          Use the CameraInfo for the intrinsics
          {!camera.available && camera.reason ? <em> — {camera.reason}</em> : null}
        </span>
      </label>
    </fieldset>
  );
}
