// Keeps the editor's config reading the deck's input: an MCAP binds
// `source.mcap`/`source.topic` (its first supported topic unless the config
// already names one of its topics); switching to a video takes the MCAP keys,
// and the CameraInfo/Rectification choices that need them, out again.
import { useEffect, useRef } from "react";
import { isMcap, type VideoRecord } from "../api/video";
import type { useEditor } from "../editor/yamlState";
import { bindMcap, needsBinding, unbindMcap } from "./mcapCamera";

type Editor = ReturnType<typeof useEditor>;

export function useMcapBinding(current: VideoRecord | null, editor: Editor) {
  const last = useRef<string | null>(null);
  const config = editor.state.pipeline.config;
  const { loaded, updateConfig } = editor;
  useEffect(() => {
    const id = current?.id ?? null;
    const changed = id !== last.current;
    last.current = id;
    if (!loaded || !current) return;
    if (isMcap(current)) {
      if (needsBinding(config, current)) updateConfig((c) => bindMcap(c, current));
    } else if (changed) {
      // Only on a switch: a reopened MCAP run's config stays while a video is loaded.
      updateConfig(unbindMcap);
    }
  }, [current, loaded, config, updateConfig]);
}
