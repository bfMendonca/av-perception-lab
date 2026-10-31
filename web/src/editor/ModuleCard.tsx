// One slot of a lane: an adapter spec, a config section, or an empty slot.
import type { ReactNode } from "react";
import type { CardReadiness } from "../models/cardReadiness";
import "../models/statusStyles.css";
import { Icon } from "../ui/icons";

export interface ModuleCardProps {
  title: string;
  subtitle?: string;
  /** Small tag next to the title (e.g. the instance name). */
  tag?: string;
  /** Long description, shown as a tooltip. */
  description?: string;
  selected?: boolean;
  errorCount?: number;
  /** `unknown`: the module type is not in this server's catalog. */
  tone?: "default" | "unknown";
  /** Model readiness (absent: the card has no model status). */
  readiness?: CardReadiness;
  onSelect: () => void;
  children?: ReactNode;
}

export function ModuleCard({ title, subtitle, tag, description, selected, errorCount = 0, tone = "default", readiness, onSelect }: ModuleCardProps) {
  const r = readiness;
  const flagged = r && (r.tone === "notready" || r.tone === "failed");
  return (
    <button
      type="button"
      className={`card${selected ? " card--selected" : ""}${errorCount ? " card--invalid" : ""}${tone === "unknown" ? " card--unknown" : ""}${r ? ` card--model card--${r.tone}` : ""}`}
      aria-pressed={selected ? true : false}
      title={description || undefined}
      data-model-state={r?.state}
      onClick={onSelect}
    >
      {flagged ? (
        <span className="card__alert" role="img" aria-label={r.alertLabel} title={r.detail ?? r.label}>
          <svg width="12" height="12" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
            <path d="M12 6.5v7.5M12 18v.01" fill="none" stroke="currentColor" strokeWidth="3.4" strokeLinecap="round" />
          </svg>
        </span>
      ) : null}
      <span className="card__title">
        <span className="card__name">{title}</span>
        {tag ? <span className="tag">{tag}</span> : null}
        {errorCount ? (
          <span className="card__err" aria-label={`${errorCount} problem${errorCount > 1 ? "s" : ""}`}>
            <Icon name="alert" size={13} />
            {errorCount}
          </span>
        ) : null}
      </span>
      {subtitle ? <span className="card__sub">{subtitle}</span> : null}
      {tone === "unknown" ? <span className="card__sub card__sub--warn">not in this server's catalog</span> : null}
      {r && r.tone !== "ready" ? (
        <span className={`card__state card__state--${r.tone}`}>
          <span className="card__pill">{r.label}</span>
          {r.summary ? <span className="card__reason">{r.summary}</span> : null}
        </span>
      ) : null}
      {r?.progress ? (
        <span
          className={`card__progress${r.progress.percent === null ? " card__progress--indeterminate" : ""}`}
          role="progressbar"
          aria-label={`Preparing ${title}`}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={r.progress.percent ?? undefined}
          aria-valuetext={r.progress.text}
        >
          <span style={{ width: `${r.progress.percent ?? 35}%` }} />
        </span>
      ) : null}
    </button>
  );
}

/** A dashed placeholder for an empty slot. */
export function EmptySlot({ label, hint }: { label: string; hint?: string }) {
  return (
    <div className="card card--empty" aria-hidden={hint ? undefined : true}>
      <span className="card__name">{label}</span>
      {hint ? <span className="card__sub">{hint}</span> : null}
    </div>
  );
}
