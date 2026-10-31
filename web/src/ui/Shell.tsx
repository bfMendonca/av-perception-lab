// App frame: navigation rail, top bar and the screen. Later views (video,
// runs) plug into the rail and the top bar.
import type { ReactNode } from "react";
import { Icon } from "./icons";

export interface NavItem {
  id: string;
  label: string;
  icon: string;
  active?: boolean;
  disabled?: boolean;
  hint?: string;
  onSelect?: () => void;
}

export interface ShellProps {
  title: string;
  nav: NavItem[];
  /** Chips next to the title (file name, status). */
  status?: ReactNode;
  /** Buttons at the right of the top bar. */
  actions?: ReactNode;
  /** Non-blocking banners under the top bar. */
  banners?: ReactNode;
  children: ReactNode;
}

export function Shell({ title, nav, status, actions, banners, children }: ShellProps) {
  return (
    <div className="shell">
      <nav className="rail" aria-label="Views">
        <div className="rail__logo" title="Perception Studio">
          <svg viewBox="0 0 32 32" width="34" height="34" aria-hidden="true">
            <defs>
              <linearGradient id="logo-g" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0" stopColor="var(--accent)" />
                <stop offset="1" stopColor="var(--accent-2)" />
              </linearGradient>
            </defs>
            <rect width="32" height="32" rx="9" fill="url(#logo-g)" />
            <path d="M9 21l5-10 4 7 2-3 3 6" fill="none" stroke="#06231f" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </div>
        {nav.map((item) => (
          <button
            key={item.id}
            type="button"
            className={`rail__item${item.active ? " rail__item--on" : ""}`}
            aria-current={item.active ? "page" : undefined}
            disabled={item.disabled}
            title={item.hint ?? item.label}
            onClick={item.onSelect}
          >
            <Icon name={item.icon} size={19} />
            <span>{item.label}</span>
          </button>
        ))}
      </nav>
      <div className="main">
        <header className="topbar">
          <h1 className="topbar__title">{title}</h1>
          <div className="topbar__status">{status}</div>
          <div className="topbar__spacer" />
          <div className="topbar__actions">{actions}</div>
        </header>
        {banners}
        <main className="screen">{children}</main>
      </div>
    </div>
  );
}
