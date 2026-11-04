// Runs screen: history list on the left, the selected run (B) on the right,
// optionally compared with another run (A). Both ids live in the hash route
// (`#/runs/<B>/vs/<A>`), so a comparison is a deep link.
import { useCallback, useEffect, useState } from "react";
import type { RunsApi } from "../api/runs";
import type { RunRecord, VideoApi } from "../api/video";
import { Icon } from "../ui/icons";
import { RunDetail } from "./RunDetail";
import { DeleteRunDialog } from "./DeleteRunDialog";
import { RunList } from "./RunList";

export interface RunsViewProps {
  videoApi: VideoApi;
  api: RunsApi;
  runId: string | null;
  compareId: string | null;
  onRoute: (runId: string | null, compareId: string | null) => void;
  onReopen: (run: RunRecord) => Promise<void>;
  onRerun: (run: RunRecord) => Promise<void>;
  /** Changes when the history should be read again (e.g. a job ended). */
  refreshKey?: unknown;
}

export function RunsView({ videoApi, api, runId, compareId, onRoute, onReopen, onRerun, refreshKey }: RunsViewProps) {
  const [runs, setRuns] = useState<RunRecord[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tick, setTick] = useState(0);
  const [pendingDelete, setPendingDelete] = useState<RunRecord | null>(null);

  useEffect(() => {
    let live = true;
    videoApi
      .listRuns()
      .then((r) => {
        if (!live) return;
        setRuns(r);
        setError(null);
      })
      .catch((exc: unknown) => live && setError(exc instanceof Error ? exc.message : String(exc)));
    return () => {
      live = false;
    };
  }, [videoApi, runId, tick, refreshKey]);

  const refresh = useCallback(() => setTick((n) => n + 1), []);
  const run = runs?.find((r) => r.id === runId) ?? null;
  const against = compareId ? (runs?.find((r) => r.id === compareId) ?? null) : null;

  return (
    <div className="runs">
      <RunList
        runs={runs}
        error={error}
        selected={runId}
        compareId={compareId}
        onSelect={(id) => onRoute(id, id === compareId ? null : compareId)}
        onCompare={(id) => onRoute(runId, id)}
        onRefresh={refresh}
        onDelete={setPendingDelete}
      />
      <section className="runs__detail" aria-label="Run">
        {!runId ? (
          <div className="state-panel">
            <Icon name="history" size={28} />
            <h2>Pick a run</h2>
            <p>Select a run on the left to see its video, summary and distances; use a run's compare button to put another one beside it.</p>
          </div>
        ) : run ? (
          <RunDetail
            run={run}
            against={against}
            runs={runs ?? []}
            api={api}
            onCompare={(id) => onRoute(runId, id)}
            onReopen={onReopen}
            onRerun={onRerun}
            onDelete={setPendingDelete}
          />
        ) : runs ? (
          <div className="state-panel" role="alert">
            <Icon name="alert" size={28} />
            <h2>Run not found</h2>
            <p>
              <code>{runId}</code> is not in the history of this server.
            </p>
          </div>
        ) : (
          <p className="muted runs__loading">Loading…</p>
        )}
      </section>
      {pendingDelete ? (
        <DeleteRunDialog
          open
          run={pendingDelete}
          api={api}
          onCancel={() => setPendingDelete(null)}
          onDeleted={() => {
            const gone = pendingDelete.id;
            setPendingDelete(null);
            if (gone === runId) onRoute(null, null);
            else if (gone === compareId) onRoute(runId, null);
            refresh();
          }}
        />
      ) : null}
    </div>
  );
}
