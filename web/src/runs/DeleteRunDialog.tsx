// Confirms the permanent deletion of one run: shows how much disk it frees
// (a `job-<id>` entry has no folder, so no size) and keeps the server's error
// (409 while the run is queued or running) in the dialog.
import { useEffect, useState } from "react";
import { ApiError } from "../api/client";
import type { RunsApi } from "../api/runs";
import type { RunRecord } from "../api/video";
import { Dialog } from "../ui/Dialog";
import { formatBytes } from "./format";
import { hasFiles, runTitle } from "./RunList";

export interface DeleteRunDialogProps {
  open: boolean;
  run: RunRecord;
  api: RunsApi;
  onDeleted: () => void;
  onCancel: () => void;
}

export function DeleteRunDialog({ open, run, api, onDeleted, onCancel }: DeleteRunDialogProps) {
  const [bytes, setBytes] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!open) return;
    setBytes(null);
    setError(null);
    setBusy(false);
    if (!hasFiles(run)) return;
    let live = true;
    api
      .size(run.id)
      .then((n) => live && setBytes(n))
      .catch(() => undefined); // the size is informative only
    return () => {
      live = false;
    };
  }, [open, run.id, api]);

  const confirm = async () => {
    if (busy) return;
    setBusy(true);
    setError(null);
    try {
      await api.remove(run.id);
      onDeleted();
    } catch (exc) {
      if (exc instanceof ApiError && exc.status === 404) {
        onDeleted(); // already gone: the list refreshes
        return;
      }
      setError(exc instanceof Error ? exc.message : String(exc));
      setBusy(false);
    }
  };

  return (
    <Dialog
      open={open}
      title="Delete run"
      onClose={() => (busy ? undefined : onCancel())}
      actions={
        <>
          <button type="button" className="btn" disabled={busy} onClick={onCancel}>
            Cancel
          </button>
          <button type="button" className="btn btn--danger" disabled={busy} onClick={() => void confirm()}>
            Delete
          </button>
        </>
      }
    >
      <p>
        Delete {runTitle(run)} permanently?{bytes !== null ? ` This frees ${formatBytes(bytes)}.` : ""}
      </p>
      {error ? (
        <p className="preset-dialog__error" role="alert">
          {error}
        </p>
      ) : null}
    </Dialog>
  );
}
