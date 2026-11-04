// Asks for a preset name. The server decides: a 409 (name taken, user preset) offers
// Overwrite or another name, any other error (the 422 for a name without
// letters or digits or one of a built-in) is shown and the dialog stays open.
import { useEffect, useState } from "react";
import { Dialog } from "../ui/Dialog";

export interface SavePresetDialogProps {
  open: boolean;
  /** Names of the user's presets (a hint only: the server is the authority). */
  existingNames: string[];
  /** Rejects with the server error (`status` 409 = the name exists). */
  onSave: (name: string, overwrite: boolean) => Promise<void>;
  onCancel: () => void;
}

export function SavePresetDialog({ open, existingNames, onSave, onCancel }: SavePresetDialogProps) {
  const [name, setName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [conflict, setConflict] = useState(false);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (open) {
      setName("");
      setError(null);
      setConflict(false);
      setBusy(false);
    }
  }, [open]);

  const submit = async (overwrite: boolean) => {
    if (busy) return;
    setBusy(true);
    setError(null);
    try {
      await onSave(name, overwrite);
    } catch (exc) {
      setConflict((exc as { status?: number }).status === 409);
      setError(exc instanceof Error ? exc.message : String(exc));
    } finally {
      setBusy(false);
    }
  };

  const same = existingNames.some((n) => n.trim().toLowerCase() === name.trim().toLowerCase()) && name.trim() !== "";

  return (
    <Dialog
      open={open}
      title="Save as preset"
      onClose={onCancel}
      actions={
        <>
          <button type="button" className="btn" onClick={onCancel}>
            Cancel
          </button>
          {conflict ? (
            <button type="button" className="btn btn--primary" disabled={busy} onClick={() => void submit(true)}>
              Overwrite
            </button>
          ) : (
            <button type="button" className="btn btn--primary" disabled={busy || name.trim() === ""} onClick={() => void submit(false)}>
              Save
            </button>
          )}
        </>
      }
    >
      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (!conflict && name.trim() !== "") void submit(false);
        }}
      >
        <label htmlFor="preset-name">Name</label>
        <input
          id="preset-name"
          className="input"
          autoFocus
          value={name}
          aria-invalid={error !== null}
          onChange={(e) => {
            setName(e.target.value);
            setConflict(false);
            setError(null);
          }}
        />
      </form>
      {same && !error ? <p className="preset-dialog__hint">You already have a preset with this name.</p> : null}
      {error ? (
        <p className="preset-dialog__error" role="alert">
          {error}
          {conflict ? " Choose Overwrite, or type another name." : ""}
        </p>
      ) : null}
    </Dialog>
  );
}
