// Composition root of the UI: loads the catalog, wires the editor store to the
// shell (open/save YAML, validation status, banners) and the editor screen.
import { useCallback, useEffect, useRef, useState } from "react";
import type { ApiClient } from "./api/client";
import type { Catalog, ValidationError } from "./api/types";
import { SUPPORTED_CATALOG_VERSION, stageByKey } from "./editor/catalog";
import { EditorView } from "./editor/EditorView";
import { createEditorStore, EditorProvider, useEditor, type Draft, type EditorStore, type Selection } from "./editor/yamlState";
import { Dialog } from "./ui/Dialog";
import { Icon } from "./ui/icons";
import { Shell } from "./ui/Shell";

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
  store?: EditorStore;
  /** Keep the working pipeline in localStorage across reloads (off in tests). */
  persistDraft?: boolean;
}

type CatalogState = { status: "loading" } | { status: "error"; message: string } | { status: "ready"; catalog: Catalog };

export function App({ client, store: given, persistDraft = false }: AppProps) {
  const [store] = useState(() => given ?? createEditorStore(client));
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
      <Studio client={client} catalog={catalog} onRetry={() => setAttempt((n) => n + 1)} />
    </EditorProvider>
  );
}

function Studio({ client, catalog, onRetry }: { client: ApiClient; catalog: CatalogState; onRetry: () => void }) {
  const editor = useEditor();
  const fileInput = useRef<HTMLInputElement>(null);
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
        const file = e.dataTransfer.files[0];
        if (file && ready) void openFile(file);
      }}
    >
      <Shell
        title="Pipeline editor"
        nav={[
          { id: "editor", label: "Editor", icon: "layers", active: true },
          { id: "runs", label: "Runs", icon: "history", disabled: true, hint: "Runs and comparison: coming soon" },
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
            <button type="button" className="btn" disabled={!editor.loaded || saving} onClick={requestSave} title="Save YAML (Ctrl+S)">
              <Icon name="save" size={15} /> {saving ? "Saving…" : "Save YAML"}
            </button>
            <button type="button" className="btn btn--primary" disabled title="Processing a video segment arrives with the video panel">
              <Icon name="play" size={14} /> Process
            </button>
          </>
        }
        banners={banners}
      >
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
        {ready ? <EditorView onOpen={() => fileInput.current?.click()} /> : null}
      </Shell>
      {dragging ? (
        <div className="dropzone" aria-hidden="true">
          <div>
            <Icon name="file" size={28} />
            <p>Drop a YAML file to open it</p>
          </div>
        </div>
      ) : null}
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
