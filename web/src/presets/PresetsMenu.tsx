// Popover body of the "Presets" button: built-in and user presets in two
// groups (name + one-line description); only user rows can be deleted.
import type { PresetInfo } from "../api/presets";
import { Icon } from "../ui/icons";
import "./presets.css";

export interface PresetsMenuProps {
  presets: PresetInfo[];
  status: "loading" | "ready" | "error";
  error: string | null;
  onPick: (id: string) => void;
  onDelete: (id: string) => void;
  onRetry: () => void;
}

function Group({ title, items, onPick, onDelete }: { title: string; items: PresetInfo[]; onPick: (id: string) => void; onDelete: (id: string) => void }) {
  return (
    <div className="presets-menu__group" role="group" aria-label={title}>
      <h3 className="presets-menu__title">{title}</h3>
      {items.length === 0 ? <p className="presets-menu__empty">{title === "Mine" ? "No saved presets yet." : "None."}</p> : null}
      <ul>
        {items.map((p) => (
          <li key={p.id} className="presets-menu__row">
            <button type="button" className="presets-menu__pick" onClick={() => onPick(p.id)}>
              <strong>{p.name}</strong>
              {p.description ? <span className="presets-menu__desc">{p.description}</span> : null}
            </button>
            {p.kind === "user" ? (
              <button type="button" className="icon-btn" aria-label={`Delete preset ${p.name}`} title="Delete" onClick={() => onDelete(p.id)}>
                <Icon name="trash" size={14} />
              </button>
            ) : null}
          </li>
        ))}
      </ul>
    </div>
  );
}

export function PresetsMenu({ presets, status, error, onPick, onDelete, onRetry }: PresetsMenuProps) {
  return (
    <div className="popover presets-menu" role="dialog" aria-label="Presets">
      {status === "error" ? (
        <div className="presets-menu__error" role="alert">
          <Icon name="alert" size={14} />
          <span>Could not list the presets: {error}</span>
          <button type="button" className="btn" onClick={onRetry}>
            Retry
          </button>
        </div>
      ) : status === "loading" ? (
        <p className="presets-menu__empty">Loading…</p>
      ) : (
        <>
          <Group title="Built-in" items={presets.filter((p) => p.kind === "builtin")} onPick={onPick} onDelete={onDelete} />
          <Group title="Mine" items={presets.filter((p) => p.kind === "user")} onPick={onPick} onDelete={onDelete} />
        </>
      )}
    </div>
  );
}
