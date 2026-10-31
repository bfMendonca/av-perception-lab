// Changes of the pipeline against the last opened or saved YAML.
import YAML from "yaml";
import { useEditor } from "./yamlState";

const show = (v: unknown) => (v === undefined ? "(absent)" : YAML.stringify(v, { lineWidth: 0, collectionStyle: "flow" }).trim());

export function DiffView() {
  const editor = useEditor();
  const entries = editor.loaded ? editor.diff() : [];
  return (
    <section className="diff" aria-label="YAML changes">
      <header className="diff__head">
        <h2>Changes</h2>
        <span className="diff__count">{entries.length ? `${entries.length} change${entries.length > 1 ? "s" : ""}` : "none"}</span>
      </header>
      {entries.length === 0 ? (
        <p className="diff__empty">{editor.loaded ? `No changes since ${editor.fileName ?? "the pipeline"} was ${editor.fileName ? "opened or saved" : "started"}.` : "Changes to the YAML show up here."}</p>
      ) : (
        <pre className="diff__body" tabIndex={0}>
          {entries.map((e) => (
            <span className="diff__entry" key={e.path}>
              {e.before !== undefined ? <span className="diff__del">{`- ${e.path}: ${show(e.before)}`}</span> : null}
              {e.after !== undefined ? <span className="diff__add">{`+ ${e.path}: ${show(e.after)}`}</span> : null}
            </span>
          ))}
        </pre>
      )}
    </section>
  );
}
