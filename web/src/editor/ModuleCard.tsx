// One slot of a lane: an adapter spec, a config section, or an empty slot.
import type { ReactNode } from "react";
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
  onSelect: () => void;
  children?: ReactNode;
}

export function ModuleCard({ title, subtitle, tag, description, selected, errorCount = 0, tone = "default", onSelect }: ModuleCardProps) {
  return (
    <button
      type="button"
      className={`card${selected ? " card--selected" : ""}${errorCount ? " card--invalid" : ""}${tone === "unknown" ? " card--unknown" : ""}`}
      aria-pressed={selected ? true : false}
      title={description || undefined}
      onClick={onSelect}
    >
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
