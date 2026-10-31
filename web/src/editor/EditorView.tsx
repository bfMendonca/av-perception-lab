// The editor screen: lanes on the canvas, the inspector and the diff on the right.
import type { ReactNode } from "react";
import { Icon } from "../ui/icons";
import { DiffView } from "./DiffView";
import { Inspector } from "./Inspector";
import { Lanes } from "./Lanes";
import { useEditor } from "./yamlState";

export function EditorView({ onOpen }: { onOpen: () => void }) {
  const editor = useEditor();
  const generalOpen = editor.selection?.kind === "general";
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
