// Banner above the lanes: how many models need preparing and how much to
// download, with Prepare all (confirmed in a dialog that lists what will be
// downloaded, its size and licences, and the files the user must provide);
// overall progress while preparing; a retry when the status could not be
// checked. Hidden when every model is ready.
import { useState } from "react";
import { Dialog } from "../ui/Dialog";
import { Icon } from "../ui/icons";
import { formatBytes } from "../video/DropZone";
import "./statusStyles.css";
import { isUserProvided, keyOf, plural, preparable, splitList, stateLabel, toneOf, where } from "./readiness";
import { useReadiness, type ModelView } from "./useModelStatus";

export interface ReadinessBannerProps {
  /** Show a model in the inspector (rows without a card too). */
  onShow?: (row: ModelView) => void;
}

export function ReadinessBanner({ onShow }: ReadinessBannerProps) {
  const readiness = useReadiness();
  const [open, setOpen] = useState(false);
  const [confirming, setConfirming] = useState(false);
  if (!readiness) return null;
  // Shown above the rest, even when every model is ready.
  const restart = readiness.restartRequired ? (
    <div className="readiness readiness--muted" role="status" aria-label="Restart required">
      <Icon name="info" size={16} />
      <div className="readiness__text">
        <strong>Packages were installed on the server</strong>
        <span className="readiness__sub">New runs use them (each run starts a fresh worker); restart the server only if it should use an upgraded package it had already imported.</span>
      </div>
    </div>
  ) : null;

  if (readiness.status === "unavailable") {
    return (
      <>
      {restart}
      <div className="readiness readiness--muted" role="status" aria-label="Model readiness">
        <Icon name="alert" size={16} />
        <div className="readiness__text">
          <strong>Could not check whether the models are ready</strong>
          {readiness.message ? <span className="readiness__sub">{readiness.message}</span> : null}
        </div>
        <button type="button" className="btn btn--sm" onClick={readiness.refresh}>
          Check again
        </button>
      </div>
      </>
    );
  }

  const { notReady, preparing, toPrepare, size } = readiness;
  if (!notReady.length) return restart;

  if (preparing.length) {
    // A model still queued counts with its declared size; one without any size makes the bar indeterminate.
    const totals = preparing.map((r) => r.progress?.total ?? r.size_bytes);
    const done = preparing.reduce((s, r) => s + (r.progress?.done ?? 0), 0);
    const total = totals.reduce<number>((s, t) => s + (t ?? 0), 0);
    const percent = total > 0 && totals.every((t) => t !== null) ? Math.min(100, Math.round((done / total) * 100)) : null;
    return (
      <>
      {restart}
      <div className="readiness readiness--preparing" role="status" aria-label="Model readiness">
        <span className="spinner" aria-hidden="true" />
        <div className="readiness__text">
          <strong>Preparing {plural(preparing.length, "model")}…</strong>
          <span className="readiness__sub">Downloads run on the server; you can keep editing. Process unlocks when every model is ready.</span>
          <div
            className={`progress progress--model readiness__bar${percent === null ? " progress--indeterminate" : ""}`}
            role="progressbar"
            aria-label="Prepare progress"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={percent ?? undefined}
          >
            <span style={{ width: `${percent ?? 35}%` }} />
          </div>
        </div>
        {readiness.busy ? (
          <button type="button" className="btn btn--sm" onClick={() => void readiness.cancel()}>
            Cancel
          </button>
        ) : null}
      </div>
      </>
    );
  }

  const failed = notReady.filter((r) => r.state === "failed");
  // Count and size cover only what Prepare all can fetch (I6); the others are named below.
  const sizeText = size.bytes ? `${size.text} to download` : null;
  const blocked = notReady.length - toPrepare.length;
  const licences = [...new Set(toPrepare.flatMap((r) => splitList(r.license)))];
  return (
    <>
    {restart}
    <div className={`readiness${failed.length ? " readiness--failed" : ""}`} role="region" aria-label="Model readiness">
      <span className="readiness__icon" aria-hidden="true">
        <Icon name="alert" size={16} />
      </span>
      <div className="readiness__text">
        <strong>
          {toPrepare.length ? plural(toPrepare.length, "model needs", "models need") + " preparing" : `${plural(notReady.length, "model is", "models are")} not ready`}
          {toPrepare.length && sizeText ? <span className="readiness__size"> · {sizeText}</span> : null}
        </strong>
        <span className="readiness__sub">
          {failed.length ? `${plural(failed.length, "model")} failed to prepare. ` : ""}
          {blocked ? `${plural(blocked, "model")} cannot be prepared from here (see the inspector). ` : ""}
          {licences.length ? `Licences: ${licences.join(", ")}. ` : ""}
          Process unlocks when every model is ready.{" "}
          <button type="button" className="linkish" aria-expanded={open} onClick={() => setOpen(!open)}>
            {open ? "Hide models" : "Show models"}
          </button>
        </span>
        {open ? (
          <ul className="readiness__list" aria-label="Models that are not ready">
            {notReady.map((r) => (
              <li key={`${r.stage}|${r.index ?? ""}`}>
                <button type="button" onClick={() => onShow?.(r)} disabled={!onShow}>
                  <span className={`reqs__pill reqs__pill--${toneOf(r.state)}`}>{stateLabel(r.state)}</span>
                  <code>{where(r)}</code>
                  <span className="readiness__name">{r.name}</span>
                  {!preparable(r) ? <span className="readiness__why">not preparable here</span> : null}
                </button>
              </li>
            ))}
          </ul>
        ) : null}
      </div>
      <button
        type="button"
        className="btn btn--warn"
        disabled={!toPrepare.length || readiness.submitting}
        title={
          toPrepare.length
            ? `Download what ${plural(toPrepare.length, "model")} need${toPrepare.length === 1 ? "s" : ""}${sizeText ? ` (${sizeText})` : ""}${licences.length ? `; licences: ${licences.join(", ")}` : ""}`
            : "None of these models can be prepared from the web"
        }
        onClick={() => setConfirming(true)}
      >
        {readiness.submitting ? <span className="spinner spinner--ink" aria-hidden="true" /> : <Icon name="open" size={14} />}
        Prepare all
      </button>
    </div>
    <PrepareAllDialog
      open={confirming}
      rows={toPrepare}
      manual={notReady.filter((r) => isUserProvided(r))}
      total={size.text}
      onClose={() => setConfirming(false)}
      onConfirm={() => {
        setConfirming(false);
        void readiness.prepare("all");
      }}
    />
    </>
  );
}

/** Confirmation of Prepare all: what is downloaded (source, size, licence), the total, and the files only the user can provide. */
function PrepareAllDialog({ open, rows, manual, total, onClose, onConfirm }: { open: boolean; rows: ModelView[]; manual: ModelView[]; total: string; onClose: () => void; onConfirm: () => void }) {
  return (
    <Dialog
      open={open}
      title="Prepare all"
      onClose={onClose}
      actions={
        <>
          <button type="button" className="btn" onClick={onClose}>
            Cancel
          </button>
          <button type="button" className="btn btn--warn" onClick={onConfirm}>
            Download
          </button>
        </>
      }
    >
      <p>The server downloads, into its weights root:</p>
      <ul className="prepare-list" aria-label="Downloads">
        {rows.map((r) => {
          const licences = splitList(r.license);
          return (
            <li key={keyOf(r)}>
              <strong>{r.name}</strong> <code>{where(r)}</code>
              <span className="prepare-list__facts">
                {r.source ? <span>{r.source}</span> : null}
                <span>{r.size_bytes ? formatBytes(r.size_bytes) : "size not declared"}</span>
                <span>{licences.length ? `licence: ${licences.join(", ")}` : "licence not declared"}</span>
              </span>
            </li>
          );
        })}
      </ul>
      <p>
        Total: <strong>{total}</strong>. By downloading you accept the licences listed.
      </p>
      {manual.length ? (
        <div className="prepare-manual" role="note" aria-label="Files you must provide">
          <p>You must provide these files yourself (they are not downloaded):</p>
          <ul>
            {manual.map((r) => (
              <li key={keyOf(r)}>
                <strong>{r.name}</strong> <code>{where(r)}</code>
                {r.message ? <span className="prepare-manual__hint">{r.message}</span> : null}
              </li>
            ))}
          </ul>
        </div>
      ) : null}
    </Dialog>
  );
}
