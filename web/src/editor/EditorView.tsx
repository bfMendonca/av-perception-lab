// The editor screen: lanes on the canvas, the inspector and the diff on the right.
import type { ReactNode } from "react";
import { ReadinessBanner } from "../models/ReadinessBanner";
import type { ModelView } from "../models/useModelStatus";
import { Icon } from "../ui/icons";
import { stageByKey } from "./catalog";
import { DiffView } from "./DiffView";
import { Inspector } from "./Inspector";
import { Lanes } from "./Lanes";
import { useEditor, type Selection } from "./yamlState";

export function EditorView({ onOpen }: { onOpen: () => void }) {
  const editor = useEditor();
  const generalOpen = editor.selection?.kind === "general";
  // A model row -> the card (or general settings tab) that configures it.
  const show = (row: ModelView) => {
    const stage = stageByKey(editor.catalog, row.stage) ?? stageByKey(editor.catalog, row.stage.split(".")[0]);
    if (!stage) return;
    const selection: Selection = stage.general
      ? { kind: "general", stageKey: stage.yaml_key }
      : { kind: "slot", stageKey: stage.yaml_key, index: stage.yaml_key === row.stage ? row.index : null };
    editor.select(selection);
  };
  return (
    <div className="editor">
      <div className="canvas" onClick={(e) => e.target === e.currentTarget && editor.select(null)}>
        {editor.loaded ? (
          <>
            <div className="canvas__bar">
              <p className="canvas__hint">Stages run left to right. Click a module to configure it; modules come from the server's adapter registry.</p>
              <button type="button" className={`btn btn--sm${generalOpen ? " btn--active" : ""}`} aria-pressed={generalOpen} onClick={() => editor.select(generalOpen ? null : { kind: "general" })}>
                <Icon name="gear" size={15} /> General settings
              </button>
            </div>
            <ReadinessBanner onShow={show} />
            <div className="canvas__scroll">
              <Lanes />
            </div>
          </>
        ) : (
          <Welcome onOpen={onOpen} onEmpty={() => editor.startEmpty()} />
        )}
      </div>
      <aside className="side" aria-label="Inspector">
        <div className="side__scroll">
          <Inspector />
        </div>
        <DiffView />
      </aside>
    </div>
  );
}

function Welcome({ onOpen, onEmpty }: { onOpen: () => void; onEmpty: () => void }): ReactNode {
  return (
    <div className="welcome">
      <div className="welcome__art" aria-hidden="true">
        {[0, 1, 2, 3].map((i) => (
          <span key={i} style={{ animationDelay: `${i * 120}ms` }} />
        ))}
      </div>
      <h2>Build a perception pipeline</h2>
      <p>Open a run configuration (YAML) to edit it visually, or drop a file anywhere on this page.</p>
      <div className="welcome__actions">
        <button type="button" className="btn btn--primary" onClick={onOpen}>
          <Icon name="upload" size={15} /> Open YAML
        </button>
        <button type="button" className="btn" onClick={onEmpty}>
          Start empty
        </button>
      </div>
    </div>
  );
}
