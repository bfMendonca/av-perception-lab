// Inspector section "Weights & requirements": what a model needs, what is
// missing, where files go and the action that fixes it. Everything comes from
// the status rows and the module schema (path fields for the override).
import { useState } from "react";
import type { PackagePlan } from "../api/models";
import type { Field } from "../api/types";
import { formatBytes } from "../video/DropZone";
import { Dialog } from "../ui/Dialog";
import { Icon } from "../ui/icons";
import "./statusStyles.css";
import { isUnknownSource, locations, MODEL_ENVIRONMENTS_DOC, onlyPackages, packagesPlan, prepareCommand, preparable, ROOT_LABEL, splitList, stateLabel, toneOf, where } from "./readiness";
import { useReadiness, type ModelView } from "./useModelStatus";

export interface ModelRequirementsProps {
  rows: ModelView[];
  /** Schema of the module: its path fields are the location override. */
  fields?: Field[];
  /** Name of the open YAML, for the CLI command. */
  fileName?: string | null;
}

const LAYER_LABEL = (key: string) => key.charAt(0).toUpperCase() + key.slice(1);

/** The section, or nothing when the module has no model status. */
export function ModelRequirements({ rows, fields = [], fileName = null }: ModelRequirementsProps) {
  const readiness = useReadiness();
  if (!rows.length) return null;
  // An override is a path option left unset by default (the adapter then picks the
  // location); one named like a missing item (asset key) is the likely one.
  const unset = fields.filter((f) => f.type === "path" && (f.default === null || f.default === undefined));
  const named = unset.filter((f) => rows.some((r) => r.missing.includes(f.name)));
  const pathFields = named.length ? named : unset;
  return (
    <section className="reqs" aria-label="Weights & requirements">
      <h3 className="insp__section">Weights &amp; requirements</h3>
      {readiness?.restartRequired ? <RestartNote /> : null}
      {rows.map((row) => (
        <ModelBlock key={`${row.stage}|${row.index ?? ""}`} row={row} single={rows.length === 1} fileName={fileName} />
      ))}
      {rows.some((r) => r.state !== "ready") ? (
        <>
          <p className="reqs__note">
            <Icon name="info" size={14} />
            {pathFields.length ? (
              <span>
                Using your own file? Set {pathFields.map((f, i) => (
                  <span key={f.name}>
                    {i ? (i === pathFields.length - 1 ? " or " : ", ") : null}
                    <code>{f.name}</code>
                  </span>
                ))}{" "}
                under Options to override the default location.
              </span>
            ) : (
              <span>Using your own files? A path option of this module, when it has one, overrides the default location.</span>
            )}
          </p>
          <p className="reqs__note reqs__note--quiet">Sources are declared by the model's code, not typed here.</p>
        </>
      ) : null}
    </section>
  );
}

function ModelBlock({ row, single, fileName }: { row: ModelView; single: boolean; fileName: string | null }) {
  const readiness = useReadiness();
  const tone = toneOf(row.state);
  const unknownSource = isUnknownSource(row);
  const packagesOnly = onlyPackages(row);
  const plan = packagesPlan(row);
  const [confirming, setConfirming] = useState(false);
  const installing = tone === "preparing" && row.progress?.kind === "packages";
  const canPrepare = readiness !== null && preparable(row);
  // The server's weights-root-relative targets; older servers: paths named in the message.
  const places = row.target ? splitList(row.target).map((t) => `${ROOT_LABEL}/${t}`) : locations(row.serverMessage ?? row.message);
  const sources = splitList(row.source);
  const licences = splitList(row.license);
  const failedNow = row.state === "failed";
  const p = row.progress;

  return (
    <div className={`reqs__model reqs__model--${tone}`} data-model-state={row.state}>
      <div className="reqs__head">
        {single ? null : <strong className="reqs__name">{row.name}</strong>}
        <code className="reqs__where">{where(row)}</code>
        <span className={`reqs__pill reqs__pill--${tone}`}>{stateLabel(row.state)}</span>
      </div>

      {tone === "ready" ? <p className="reqs__ok">Every requirement is present: nothing to prepare.</p> : null}

      {tone === "preparing" ? (
        <div className="reqs__progress">
          <div
            className={`progress progress--model${p?.fraction == null ? " progress--indeterminate" : ""}`}
            role="progressbar"
            aria-label={`Preparing ${row.name}`}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={p?.fraction == null ? undefined : Math.round(p.fraction * 100)}
          >
            <span style={{ width: `${p?.fraction == null ? 35 : p.fraction * 100}%` }} />
          </div>
          <div className="reqs__progress-text">
            <span>
              {row.state === "queued"
                ? "Waiting in the prepare queue"
                : installing
                  ? "Installing packages…"
                  : p && p.total !== null
                  ? `${formatBytes(p.done)} of ${formatBytes(p.total)}`
                  : p && p.done > 0
                    ? `${formatBytes(p.done)} downloaded`
                    : "Preparing…"}
              {p?.item ? (
                <>
                  {" "}
                  <code className="reqs__item">{p.item}</code>
                </>
              ) : null}
            </span>
            {readiness?.busy ? (
              <button type="button" className="btn btn--xs btn--ghost" onClick={() => void readiness.cancel()}>
                Cancel
              </button>
            ) : null}
          </div>
          {installing && p?.output.length ? (
            <pre className="reqs__output" aria-label="uv output">
              {p.output.slice(-8).join("\n")}
            </pre>
          ) : null}
        </div>
      ) : null}

      {row.message && tone !== "ready" && tone !== "preparing" ? (
        <p className={`reqs__message reqs__message--${tone}`} role={failedNow ? "alert" : undefined}>
          {row.message}
        </p>
      ) : null}

      <dl className="reqs__facts">
        {Object.keys(row.layers).length ? (
          <div>
            <dt>Layers</dt>
            <dd className="reqs__layers">
              {Object.entries(row.layers).map(([layer, state]) => (
                <span key={layer} className={`reqs__layer reqs__layer--${state === "ready" ? "ok" : "bad"}`} title={`${LAYER_LABEL(layer)}: ${state}`}>
                  <Icon name={state === "ready" ? "check" : "alert"} size={12} />
                  {LAYER_LABEL(layer)}
                  {state === "ready" ? null : <span className="reqs__layer-state">{state}</span>}
                </span>
              ))}
            </dd>
          </div>
        ) : null}
        {row.missing.length && tone !== "ready" ? (
          <div>
            <dt>Missing</dt>
            <dd>
              {row.missing.map((m) => (
                <code key={m} className="reqs__missing">
                  {m}
                </code>
              ))}
            </dd>
          </div>
        ) : null}
        {tone !== "ready" ? (
          <div>
            <dt>Size</dt>
            <dd>{row.size_bytes ? `${formatBytes(row.size_bytes)} to download` : "not declared"}</dd>
          </div>
        ) : null}
        {sources.length ? (
          <div>
            <dt>Source</dt>
            <dd className="reqs__list">
              {sources.map((src) => (
                <span key={src}>{src}</span>
              ))}
            </dd>
          </div>
        ) : null}
        {licences.length ? (
          <div>
            <dt>Licence</dt>
            <dd>{licences.join(", ")}</dd>
          </div>
        ) : null}
        {tone === "ready" && !places.length ? null : (
        <div>
          <dt>Stored in</dt>
          <dd>
            {places.length ? (
              places.map((pl) => (
                <code key={pl} className="reqs__path">
                  {pl}
                </code>
              ))
            ) : (
              <span className="muted-text">
                the server's weights root (<code>{ROOT_LABEL}</code>)
              </span>
            )}
          </dd>
        </div>
        )}
      </dl>

      {canPrepare ? (
        <button type="button" className="btn btn--warn reqs__action" disabled={readiness.submitting} onClick={() => void readiness.prepare([{ stage: row.stage, index: row.index }])}>
          {failedNow || row.attempt ? (
            <>
              <Icon name="reset" size={14} /> Retry
            </>
          ) : (
            <>
              <Icon name="open" size={14} /> Prepare model
            </>
          )}
        </button>
      ) : null}

      {unknownSource && tone !== "preparing" ? (
        <div className="reqs__hint">
          <p>The web server only downloads sources its adapters vouch for. Prepare this model from the command line, on the server:</p>
          <CopyCommand command={prepareCommand(fileName)} />
        </div>
      ) : null}

      {plan?.action === "base" && plan.command && tone !== "preparing" ? (
        <div className="reqs__hint">
          <p>The base Python stack is not installed on the server, and the server never installs it. Run this once in the repository on the server, then restart the server:</p>
          <CopyCommand command={plan.command} />
        </div>
      ) : null}

      {plan?.action === "install" && plan.command && tone !== "preparing" && readiness ? (
        <div className="reqs__hint">
          <p>
            Its Python packages come from the extra <code>{plan.extra}</code>. A dry-run against the server's environment is clean{plan.message ? ` (${plan.message})` : ""}.
          </p>
          <button type="button" className="btn btn--warn reqs__action" disabled={readiness.submitting} onClick={() => setConfirming(true)}>
            <Icon name="open" size={14} /> Install packages
          </button>
          <InstallDialog
            open={confirming}
            plan={plan}
            onClose={() => setConfirming(false)}
            onConfirm={() => {
              setConfirming(false);
              void readiness.installPackages({ stage: row.stage, index: row.index });
            }}
          />
        </div>
      ) : null}

      {packagesOnly && !unknownSource && tone !== "preparing" && plan === undefined ? (
        <p className="reqs__note">
          <Icon name="info" size={14} />
          <span>Only Python packages are missing. Prepare downloads weights and code, not packages: install them on the server and restart it.</span>
        </p>
      ) : null}

      {row.state === "conflict" ? (
        <p className="reqs__note">
          <Icon name="info" size={14} />
          <span>
            Installing this model's packages conflicts with the current environment, which is left untouched. See <code>{MODEL_ENVIRONMENTS_DOC}</code> for the ways to resolve it.
          </span>
        </p>
      ) : null}
    </div>
  );
}

/** Packages were installed on the server since it started. */
function RestartNote() {
  return (
    <p className="reqs__note reqs__note--restart" role="status">
      <Icon name="info" size={14} />
      <span>Packages were installed on the server. Restart the server to use them.</span>
    </p>
  );
}

/** Confirmation of Install packages: what runs, where, and what happens after. */
function InstallDialog({ open, plan, onClose, onConfirm }: { open: boolean; plan: PackagePlan; onClose: () => void; onConfirm: () => void }) {
  return (
    <Dialog
      open={open}
      title="Install packages"
      onClose={onClose}
      actions={
        <>
          <button type="button" className="btn" onClick={onClose}>
            Cancel
          </button>
          <button type="button" className="btn btn--warn" onClick={onConfirm}>
            Install
          </button>
        </>
      }
    >
      <p>
        This installs the extra <code>{plan.extra}</code> into the server's Python environment by running, on the server:
      </p>
      <pre className="reqs__output">{plan.command}</pre>
      <p>Every extra already installed is named, so nothing is removed{plan.message ? `; the dry-run says: ${plan.message}` : ""}. The server runs the dry-run again first and stops if it is no longer clean.</p>
      <p>Restart the server afterwards to use the new packages.</p>
    </Dialog>
  );
}

/** A shell command with a Copy button (falls back to selecting it). */
export function CopyCommand({ command }: { command: string }) {
  const [copied, setCopied] = useState<"idle" | "copied" | "failed">("idle");
  const copy = async () => {
    try {
      if (!navigator.clipboard) throw new Error("no clipboard");
      await navigator.clipboard.writeText(command);
      setCopied("copied");
    } catch {
      setCopied("failed");
    }
    setTimeout(() => setCopied("idle"), 2000);
  };
  return (
    <div className="cmd">
      <code className="cmd__text" tabIndex={0} aria-label="Command">
        {command}
      </code>
      <button type="button" className="btn btn--xs" aria-label="Copy command" onClick={() => void copy()}>
        {copied === "copied" ? <Icon name="check" size={13} /> : <Icon name="file" size={13} />}
        {copied === "copied" ? "Copied" : copied === "failed" ? "Select and copy" : "Copy"}
      </button>
    </div>
  );
}
