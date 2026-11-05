// The Rectification lane (stage `rectify`): an on/off switch offered only for
// an MCAP input, enabled when the selected topic's CameraInfo has distortion
// (rules in mcapCamera.ts); when on, a card opens its options (alpha) in the
// inspector. Registered over the generic lane; App provides the current input
// through StageContext.
import { useContext } from "react";
import { isMcap, type VideoRecord } from "../api/video";
import { defaultsFromFields, sectionFields, summarize } from "../editor/catalog";
import { ModuleCard } from "../editor/ModuleCard";
import { isRecord, isUnder, toDotted } from "../editor/paths";
import { registerStageView, StageContext, type StageContextValue, type StageViewProps } from "../editor/registry";
import { useEditor } from "../editor/yamlState";
import { Icon } from "../ui/icons";
import { findTopic, RECTIFY_KEY, rectifyAvailability, rectifyEnabled, setRectify, sourceTopic } from "./mcapCamera";

/** StageContext key of the deck's current input (a VideoRecord or null). */
export const INPUT_CONTEXT_KEY = "input";

const inputOf = (context: StageContextValue) => (context[INPUT_CONTEXT_KEY] ?? null) as VideoRecord | null;

export function RectifyLane({ stage, value, catalog, errors, selection, select }: StageViewProps) {
  const editor = useEditor();
  const input = inputOf(useContext(StageContext));
  const config = editor.state.pipeline.config;
  const enabled = rectifyEnabled(config);
  const topic = findTopic(input, sourceTopic(config));
  const availability = isMcap(input) ? rectifyAvailability(topic) : { state: "unsupported" as const, reason: "needs an MCAP input" };
  const available = availability.state === "available";
  const fields = sectionFields(catalog, stage.yaml_key);
  const selected = selection?.kind === "slot" && selection.stageKey === stage.yaml_key;
  const ownErrors = errors.filter((e) => e.path === stage.yaml_key || e.path === `${stage.yaml_key}.enabled`);

  const toggle = () => {
    if (enabled) {
      editor.updateConfig((c) => setRectify(c, false));
      if (selected) editor.select(null);
    } else {
      editor.updateConfig((c) => setRectify(c, true, defaultsFromFields(fields ?? [])));
    }
  };

  return (
    <section className="lane" aria-label={stage.title}>
      <header className="lane__head">
        <h2 className="lane__title" title={`${stage.title} (optional)`}>
          {stage.title}
        </h2>
      </header>
      <div className="lane__body">
        <div className={`rectify${available || enabled ? "" : " rectify--off"}`}>
          <button
            type="button"
            role="switch"
            aria-checked={enabled}
            aria-label="Rectify the images"
            aria-describedby="rectify-why"
            className={`switch${enabled ? " switch--on" : ""}`}
            disabled={!enabled && !available}
            onClick={toggle}
          >
            <span className="switch__thumb" />
            <span className="switch__text">{enabled ? "On" : "Off"}</span>
          </button>
          <p id="rectify-why" className={`rectify__why${!available && enabled ? " rectify__why--warn" : ""}`}>
            {available
              ? enabled
                ? `Undistorted with ${topic?.cameraInfoTopic ?? "the CameraInfo"}`
                : "Remove the lens distortion with the topic's CameraInfo"
              : availability.reason}
          </p>
        </div>
        {enabled && isRecord(value) ? (
          <ModuleCard
            title={stage.title}
            subtitle={summarize(fields, value, 2) || "defaults"}
            selected={selected}
            errorCount={errors.filter((e) => isUnder(e.path, toDotted([stage.yaml_key]))).length}
            onSelect={() => select({ kind: "slot", stageKey: stage.yaml_key, index: null })}
          />
        ) : null}
        {ownErrors.map((e, i) => (
          <p className="lane__error" key={i}>
            <Icon name="alert" size={13} /> {e.message}
          </p>
        ))}
      </div>
    </section>
  );
}

// Offered only for an MCAP input; still shown when a YAML turned it on, so
// the setting (and the server's error about it) stays visible.
registerStageView(RECTIFY_KEY, RectifyLane, {
  hidden: (value, context) => !isMcap(inputOf(context)) && !(isRecord(value) && value.enabled === true),
});
