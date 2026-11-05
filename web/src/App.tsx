// Composition root of the UI: loads the catalog, wires the editor store to the
// shell (open/save YAML, validation status, banners), the video deck (library,
// segment, Process, jobs) and the screens (editor, runs) behind hash routes.
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ApiClient } from "./api/client";
import type { Catalog, ValidationError } from "./api/types";
import { createModelsApi, type ModelsApi } from "./api/models";
import { ApiError } from "./api/client";
import { createPresetsApi, type PresetsApi } from "./api/presets";
import { createVideoApi, type VideoApi } from "./api/video";
import { SUPPORTED_CATALOG_VERSION, stageByKey } from "./editor/catalog";
import { EditorView } from "./editor/EditorView";
import { StageContext } from "./editor/registry";
import { ReadinessProvider, useModelStatus } from "./models/useModelStatus";
import { createEditorStore, EditorProvider, useEditor, type Draft, type EditorStore, type Selection } from "./editor/yamlState";
import { Dialog } from "./ui/Dialog";
import { Icon } from "./ui/icons";
import { useHashRoute } from "./ui/route";
import { Shell } from "./ui/Shell";
import { PresetsMenu } from "./presets/PresetsMenu";
import { SavePresetDialog } from "./presets/SavePresetDialog";
import { usePresets } from "./presets/usePresets";
import { RunsView } from "./runs/RunsView";
import { createRunsApi, type RunsApi } from "./api/runs";
import type { RunRecord } from "./api/video";
import { classifyFiles, isRawH264, type DroppedFiles } from "./video/DropZone";
import { INPUT_CONTEXT_KEY } from "./video/RectifyLane"; // also registers the Rectification lane
import { useJobs } from "./video/useJobs";
import { useMcapBinding } from "./video/useMcapBinding";
import { useProcess } from "./video/useProcess";
import { useVideoSession } from "./video/useVideoSession";
import { VideoDeck } from "./video/VideoDeck";

const DRAFT_KEY = "ddd.editor.draft.v1";
const SAVE_WARNING_KEY = "ddd.editor.saveWarningSeen";

function storageGet(key: string): string | null {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}

function storageSet(key: string, value: string | null) {
  try {
    if (value === null) window.localStorage.removeItem(key);
    else window.localStorage.setItem(key, value);
  } catch {
    // storage unavailable (private mode, blocked): the feature just does not persist
  }
}

export interface AppProps {
  client: ApiClient;
  /** Video library, runs and GPU routes (default: the HTTP client). */
  videoApi?: VideoApi;
  /** Results, comparison, reopen and rerun routes (default: the HTTP client). */
  runsApi?: RunsApi;
  /** Model status and Prepare routes (default: the HTTP client). */
  modelsApi?: ModelsApi;
  /** Built-in and user presets routes (default: the HTTP client). */
  presetsApi?: PresetsApi;
  store?: EditorStore;
  /** Keep the working pipeline in localStorage across reloads (off in tests). */
  persistDraft?: boolean;
}

type CatalogState = { status: "loading" } | { status: "error"; message: string } | { status: "ready"; catalog: Catalog };

export function App({ client, videoApi, runsApi, modelsApi, presetsApi, store: given, persistDraft = false }: AppProps) {
  const [store] = useState(() => given ?? createEditorStore(client));
  const [video] = useState(() => videoApi ?? createVideoApi());
  const [runs] = useState(() => runsApi ?? createRunsApi());
  const [models] = useState(() => modelsApi ?? createModelsApi());
  const [presets] = useState(() => presetsApi ?? createPresetsApi());
  const [catalog, setCatalog] = useState<CatalogState>({ status: "loading" });
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let live = true;
    setCatalog({ status: "loading" });
    client
      .catalog()
      .then((c) => {
        if (!live) return;
        store.setCatalog(c);
        setCatalog({ status: "ready", catalog: c });
      })
      .catch((exc: unknown) => live && setCatalog({ status: "error", message: exc instanceof Error ? exc.message : String(exc) }));
    return () => {
      live = false;
    };
  }, [client, store, attempt]);

  useEffect(() => {
    if (!persistDraft) return;
    if (!store.getState().loaded) {
      const raw = storageGet(DRAFT_KEY);
      if (raw) {
        try {
          store.restore(JSON.parse(raw) as Draft);
          store.notify({ tone: "info", text: "Restored the pipeline you were editing." });
        } catch {
          storageSet(DRAFT_KEY, null);
        }
      }
    }
    let last: unknown = null;
    return store.subscribe(() => {
      const snap = store.snapshot();
      if (snap && snap.pipeline !== last) {
        last = snap.pipeline;
        storageSet(DRAFT_KEY, JSON.stringify(snap));
      }
    });
  }, [persistDraft, store]);

  return (
    <EditorProvider store={store}>
      <Studio client={client} video={video} runs={runs} models={models} presets={presets} remember={persistDraft} catalog={catalog} onRetry={() => setAttempt((n) => n + 1)} />
    </EditorProvider>
  );
}

function Studio({ client, video, runs, models, presets: presetsApi, remember, catalog, onRetry }: { client: ApiClient; video: VideoApi; runs: RunsApi; models: ModelsApi; presets: PresetsApi; remember: boolean; catalog: CatalogState; onRetry: () => void }) {
  const editor = useEditor();
  const [route, navigate] = useHashRoute();
  const session = useVideoSession(video, { remember });
  const jobs = useJobs(video, { onDone: (runId) => navigate({ view: "runs", runId }) });
  const readiness = useModelStatus(editor.loaded && catalog.status === "ready" ? editor.toConfig() : null, { api: models, debounceMs: editor.debounceMs, notify: editor.notify });
  const process = useProcess(video, session, jobs, editor, readiness);
  useMcapBinding(session.current, editor);
  const stageContext = useMemo(() => ({ [INPUT_CONTEXT_KEY]: session.current }), [session.current]);
  const fileInput = useRef<HTMLInputElement>(null);
  const presets = usePresets(presetsApi);
  const [presetsOpen, setPresetsOpen] = useState(false);
  const [presetDialog, setPresetDialog] = useState(false);
  const presetsWrap = useRef<HTMLDivElement>(null);
  const [saveDialog, setSaveDialog] = useState(false);
  const [saving, setSaving] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [bannerHidden, setBannerHidden] = useState(false);

  const openFile = useCallback(
    async (file: File) => {
      if (editor.dirty && !window.confirm(`Discard the unsaved changes and open ${file.name}?`)) return;
      try {
        await editor.load(await file.text(), file.name);
      } catch (exc) {
        editor.notify({ tone: "error", text: `Could not open ${file.name}: ${exc instanceof Error ? exc.message : String(exc)}` });
      }
    },
    [editor],
  );

  const openPreset = useCallback(
    async (id: string) => {
      const info = presets.presets.find((p) => p.id === id);
      const name = info?.name ?? id;
      if (editor.dirty && !window.confirm(`Discard the unsaved changes and open preset ${name}?`)) return;
      setPresetsOpen(false);
      try {
        editor.loadPreset(await presetsApi.get(id));
      } catch (exc) {
        editor.notify({ tone: "error", text: `Could not open preset ${name}: ${exc instanceof Error ? exc.message : String(exc)}` });
        void presets.refresh(); // a stale entry (404) disappears from the list
      }
    },
    [editor, presets, presetsApi],
  );

  const deletePreset = useCallback(
    async (id: string) => {
      const name = presets.presets.find((p) => p.id === id)?.name ?? id;
      if (!window.confirm(`Delete the preset ${name}?`)) return;
      try {
        await presetsApi.remove(id);
      } catch (exc) {
        // 404: already gone (another tab, a hand edit): the refresh below is the answer
        if (!(exc instanceof ApiError && exc.status === 404)) editor.notify({ tone: "error", text: `Could not delete preset ${name}: ${exc instanceof Error ? exc.message : String(exc)}` });
      }
      await presets.refresh();
    },
    [editor, presets, presetsApi],
  );

  const savePreset = useCallback(
    async (name: string, overwrite: boolean) => {
      await presetsApi.save(name, editor.state.pipeline.config, editor.state.pipeline.extras, overwrite); // errors stay in the dialog
      setPresetDialog(false);
      editor.notify({ tone: "info", text: `Saved preset ${name}.` });
      await presets.refresh();
    },
    [editor, presets, presetsApi],
  );

  useEffect(() => {
    if (!presetsOpen) return;
    const onDown = (e: MouseEvent) => {
      if (presetsWrap.current && !presetsWrap.current.contains(e.target as Node)) setPresetsOpen(false);
    };
    const onEsc = (e: KeyboardEvent) => e.key === "Escape" && setPresetsOpen(false);
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onEsc);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onEsc);
    };
  }, [presetsOpen]);

  const save = useCallback(async () => {
    setSaving(true);
    try {
      const text = await client.dump(editor.state.pipeline.config, editor.state.pipeline.extras);
      const name = editor.fileName ?? "pipeline.yaml";
      const url = URL.createObjectURL(new Blob([text], { type: "application/x-yaml" }));
      const a = document.createElement("a");
      a.href = url;
      a.download = name;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      editor.markSaved(name);
      editor.notify({ tone: "info", text: `Saved ${name}.` });
    } catch (exc) {
      editor.notify({ tone: "error", text: `Could not save: ${exc instanceof Error ? exc.message : String(exc)}` });
    } finally {
      setSaving(false);
    }
  }, [client, editor]);

  const reopenRun = useCallback(
    async (run: RunRecord) => {
      if (editor.dirty && !window.confirm("Discard the unsaved changes in the editor and open this run's configuration?")) return;
      try {
        const parsed = await runs.reopen(run.id);
        editor.loadParsed({ config: parsed.config, extras: parsed.extras }, parsed.file_name);
        navigate({ view: "editor" });
        editor.notify({ tone: "info", text: `Opened the configuration of ${run.video ?? run.id} (from ${parsed.origin}).` });
      } catch (exc) {
        editor.notify({ tone: "error", text: `Could not reopen the run: ${exc instanceof Error ? exc.message : String(exc)}` });
      }
    },
    [editor, navigate, runs],
  );

  const rerunRun = useCallback(
    async (run: RunRecord) => {
      try {
        const jobId = await runs.rerun(run.id);
        jobs.track(jobId, { label: `Rerun · ${run.video ?? run.id}`, expected: run.frames ?? 0 });
        editor.notify({ tone: "info", text: `Rerun queued (job ${jobId}). Its progress shows in the editor's job panel; the new run opens here when it ends.` });
      } catch (exc) {
        editor.notify({ tone: "error", text: `Could not rerun: ${exc instanceof Error ? exc.message : String(exc)}` });
      }
    },
    [editor, jobs, runs],
  );
  const finishedJobs = jobs.jobs.filter((j) => j.status === "done").length;

  const sessionWarned = useRef(false);
  const requestSave = () => {
    if (storageGet(SAVE_WARNING_KEY) || sessionWarned.current) void save();
    else setSaveDialog(true);
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "s" && editor.loaded) {
        e.preventDefault();
        requestSave();
      } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "o") {
        e.preventDefault();
        fileInput.current?.click();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  const notice = editor.notice;
  const dismissNotice = editor.dismissNotice;
  useEffect(() => {
    if (notice?.tone !== "info") return;
    const t = setTimeout(dismissNotice, 4000);
    return () => clearTimeout(t);
  }, [notice, dismissNotice]);

  const addVideo = useCallback(
    (files: DroppedFiles) => {
      if (!files.video) return;
      if (isRawH264(files.video.name) && !files.meta) {
        editor.notify({ tone: "warn", text: `${files.video.name} is a raw H.264 stream: drop it together with its .meta.json (sensor timestamps).` });
        return;
      }
      void session.uploadFile(files.video, files.meta);
    },
    [editor, session],
  );

  const ready = catalog.status === "ready";
  const newer = ready && catalog.catalog.catalog_version > SUPPORTED_CATALOG_VERSION;

  const banners = (
    <>
      {newer && !bannerHidden ? (
        <div className="banner banner--warn" role="status" aria-label="Catalog version">
          <Icon name="info" size={15} />
          <span>
            This server describes its pipeline with catalog version {catalog.catalog.catalog_version}; this interface was built for version {SUPPORTED_CATALOG_VERSION}. Everything still renders, unfamiliar parts with generic controls.
          </span>
          <button type="button" className="icon-btn" aria-label="Dismiss" onClick={() => setBannerHidden(true)}>
            <Icon name="x" />
          </button>
        </div>
      ) : null}
      {editor.notice ? (
        <div className={`toast toast--${editor.notice.tone}`} role={editor.notice.tone === "error" ? "alert" : "status"}>
          <Icon name={editor.notice.tone === "info" ? "check" : "alert"} size={15} />
          <span>{editor.notice.text}</span>
          <button type="button" className="icon-btn" aria-label="Dismiss" onClick={() => editor.dismissNotice()}>
            <Icon name="x" />
          </button>
        </div>
      ) : null}
    </>
  );

  return (
    <ReadinessProvider value={readiness}>
      <div
        className="app"
        onDragOver={(e) => {
          if (e.dataTransfer.types.includes("Files")) {
            e.preventDefault();
            setDragging(true);
          }
        }}
        onDragLeave={(e) => e.currentTarget === e.target && setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          const files = classifyFiles(Array.from(e.dataTransfer.files));
          if (files.video) {
            if (route.view !== "editor") navigate({ view: "editor" });
            addVideo(files);
          } else if (files.yaml && ready) void openFile(files.yaml);
          else if (e.dataTransfer.files.length) editor.notify({ tone: "warn", text: "Drop a video (MP4, MOV, MKV, .h264 with its .meta.json), a ROS 2 .mcap or a pipeline YAML." });
        }}
      >
        <Shell
          title={route.view === "runs" ? "Runs & comparison" : "Pipeline editor"}
          nav={[
            { id: "editor", label: "Editor", icon: "layers", active: route.view === "editor", onSelect: () => navigate({ view: "editor" }) },
            {
              id: "runs",
              label: "Runs",
              icon: "history",
              active: route.view === "runs",
              badge: jobs.jobs.filter((j) => j.status === "queued" || j.status === "running").length || undefined,
              onSelect: () => navigate(route.view === "runs" ? route : { view: "runs", runId: null }),
            },
          ]}
          status={ready ? <StatusChips /> : null}
          actions={
            <>
              <input
                ref={fileInput}
                type="file"
                accept=".yaml,.yml,application/x-yaml,text/yaml"
                hidden
                data-testid="open-yaml-input"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  e.target.value = "";
                  if (file) void openFile(file);
                }}
              />
              <button type="button" className="btn" disabled={!ready} onClick={() => fileInput.current?.click()} title="Open YAML (Ctrl+O)">
                <Icon name="upload" size={15} /> Open YAML
              </button>
              <div className="chip-wrap" ref={presetsWrap}>
                <button type="button" className="btn" disabled={!ready} aria-expanded={presetsOpen} aria-haspopup="dialog" onClick={() => {
                    if (!presetsOpen) void presets.refresh();
                    setPresetsOpen((o) => !o);
                  }} title="Load a preset pipeline">
                  <Icon name="layers" size={15} /> Presets
                </button>
                {presetsOpen ? <PresetsMenu presets={presets.presets} status={presets.status} error={presets.error} onPick={(id) => void openPreset(id)} onDelete={(id) => void deletePreset(id)} onRetry={() => void presets.refresh()} /> : null}
              </div>
              <button type="button" className="btn" disabled={!editor.loaded || saving} onClick={requestSave} title="Save YAML (Ctrl+S)">
                <Icon name="save" size={15} /> {saving ? "Saving…" : "Save YAML"}
              </button>
              <button type="button" className="btn" disabled={!editor.loaded} onClick={() => setPresetDialog(true)} title="Save the pipeline as a preset">
                <Icon name="save" size={15} /> Save as preset…
              </button>
              <button
                type="button"
                className="btn btn--primary"
                disabled={Boolean(process.blocked) || process.busy}
                title={process.blocked ?? "Process the selected segment"}
                onClick={() => void process.process(false)}
              >
                <Icon name="play" size={14} /> Process
              </button>
            </>
          }
          banners={banners}
        >
          {route.view === "runs" ? (
            <RunsView
              videoApi={video}
              api={runs}
              runId={route.runId}
              compareId={route.compareId ?? null}
              onRoute={(runId, compareId) => navigate({ view: "runs", runId, compareId })}
              onReopen={reopenRun}
              onRerun={rerunRun}
              refreshKey={finishedJobs}
            />
          ) : (
            <div className="workspace">
              <VideoDeck
                session={session}
                jobs={jobs}
                process={process}
                previewUrl={(runId) => video.runFileUrl(runId, "preview")}
                onFiles={addVideo}
                onOpenRun={(runId) => navigate({ view: "runs", runId })}
              />
              {catalog.status === "loading" ? <LoadingCanvas /> : null}
              {catalog.status === "error" ? (
                <div className="state-panel" role="alert">
                  <Icon name="alert" size={28} />
                  <h2>Could not reach the server</h2>
                  <p>
                    The catalog request failed: <code>{catalog.message}</code>
                  </p>
                  <button type="button" className="btn btn--primary" onClick={onRetry}>
                    Retry
                  </button>
                </div>
              ) : null}
              {ready ? (
                <StageContext.Provider value={stageContext}>
                  <EditorView onOpen={() => fileInput.current?.click()} />
                </StageContext.Provider>
              ) : null}
            </div>
          )}
        </Shell>
        {dragging ? (
          <div className="dropzone" aria-hidden="true">
            <div>
              <Icon name="film" size={28} />
              <p>Drop a video or an MCAP to load it, or a pipeline YAML to open it</p>
            </div>
          </div>
        ) : null}
        {process.dialog}
        <SavePresetDialog open={presetDialog} existingNames={presets.presets.filter((p) => p.kind === "user").map((p) => p.name)} onSave={savePreset} onCancel={() => setPresetDialog(false)} />
        <Dialog
          open={saveDialog}
          title="Save as regenerated YAML"
          onClose={() => setSaveDialog(false)}
          actions={
            <>
              <button type="button" className="btn" onClick={() => setSaveDialog(false)}>
                Cancel
              </button>
              <button
                type="button"
                className="btn btn--primary"
                autoFocus
                onClick={() => {
                  setSaveDialog(false);
                  sessionWarned.current = true;
                  storageSet(SAVE_WARNING_KEY, "1");
                  void save();
                }}
              >
                Save YAML
              </button>
            </>
          }
        >
          <p>The file is written from the edited configuration, so comments and the original key order are not kept. Unknown keys are saved as they are.</p>
          <p className="muted">This notice is shown once.</p>
        </Dialog>
      </div>
    </ReadinessProvider>
  );
}

function selectionForError(editor: ReturnType<typeof useEditor>, error: ValidationError): Selection {
  const match = /^([^.[]+)(?:\[(\d+)\])?/.exec(error.path);
  const stage = match ? stageByKey(editor.catalog, match[1]) : undefined;
  if (!stage) return { kind: "general", stageKey: "__other__" };
  if (stage.general) return { kind: "general", stageKey: stage.yaml_key };
  if (stage.multiple && match?.[2] !== undefined) return { kind: "slot", stageKey: stage.yaml_key, index: Number(match[2]) };
  return { kind: "slot", stageKey: stage.yaml_key, index: null };
}

function StatusChips() {
  const editor = useEditor();
  const [open, setOpen] = useState(false);
  if (!editor.loaded) return null;
  const v = editor.validation;
  return (
    <>
      <span className="chip chip--file" title={editor.fileName ?? "Not saved yet"}>
        <Icon name="file" size={13} /> {editor.fileName ?? "untitled.yaml"}
      </span>
      {editor.dirty ? <span className="chip chip--warn">Unsaved changes</span> : null}
      <div className="chip-wrap">
        <button
          type="button"
          className={`chip chip--btn chip--${v === "valid" ? "ok" : v === "invalid" ? "bad" : v === "unavailable" ? "warn" : "muted"}`}
          aria-live="polite"
          aria-expanded={v === "invalid" ? open : undefined}
          disabled={v !== "invalid"}
          title={editor.validationMessage ?? undefined}
          onClick={() => setOpen(!open)}
        >
          {v === "pending" || v === "idle" ? <span className="spinner" aria-hidden="true" /> : <Icon name={v === "valid" ? "check" : "alert"} size={13} />}
          {v === "valid" ? "Valid" : v === "invalid" ? `${editor.errors.length} problem${editor.errors.length > 1 ? "s" : ""}` : v === "unavailable" ? "Validation unavailable" : "Validating…"}
        </button>
        {open && v === "invalid" ? (
          <div className="popover" role="dialog" aria-label="Validation problems">
            <ul>
              {editor.errors.map((e) => (
                <li key={`${e.path}:${e.message}`}>
                  <button
                    type="button"
                    onClick={() => {
                      editor.select(selectionForError(editor, e));
                      setOpen(false);
                    }}
                  >
                    <code>{e.path || "config"}</code>
                    <span>{e.message}</span>
                  </button>
                </li>
              ))}
            </ul>
          </div>
        ) : null}
      </div>
    </>
  );
}

function LoadingCanvas() {
  return (
    <div className="editor" aria-busy="true" aria-label="Loading the catalog">
      <div className="canvas">
        <div className="canvas__scroll">
          <div className="lanes">
            {[0, 1, 2, 3, 4, 5].map((i) => (
              <div className="lane lane--skeleton" key={i}>
                <span className="skel skel--title" />
                <span className="skel skel--card" />
                {i % 2 ? <span className="skel skel--card" /> : null}
              </div>
            ))}
          </div>
        </div>
      </div>
      <aside className="side" />
    </div>
  );
}
