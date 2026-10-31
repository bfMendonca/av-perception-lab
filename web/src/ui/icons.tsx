// Inline stroke icons (no icon font, no network).
const PATHS: Record<string, string> = {
  plus: "M12 5v14M5 12h14",
  x: "M6 6l12 12M18 6L6 18",
  chevron: "M9 6l6 6-6 6",
  alert: "M12 8v5M12 16.5v.01M10.3 3.9L2.4 18a2 2 0 001.7 3h15.8a2 2 0 001.7-3L13.7 3.9a2 2 0 00-3.4 0z",
  check: "M5 12.5l4.5 4.5L19 7.5",
  reset: "M4 12a8 8 0 1 0 2.4-5.7M4 4v4.5h4.5",
  folder: "M3 7a2 2 0 012-2h4l2 2h8a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2z",
  gear: "M12 15a3 3 0 100-6 3 3 0 000 6zM19.4 15a1.7 1.7 0 00.3 1.8l.1.1a2 2 0 11-2.8 2.8l-.1-.1a1.7 1.7 0 00-1.8-.3 1.7 1.7 0 00-1 1.5V21a2 2 0 11-4 0v-.1a1.7 1.7 0 00-1.1-1.5 1.7 1.7 0 00-1.8.3l-.1.1a2 2 0 11-2.8-2.8l.1-.1a1.7 1.7 0 00.3-1.8 1.7 1.7 0 00-1.5-1H3a2 2 0 110-4h.1a1.7 1.7 0 001.5-1.1 1.7 1.7 0 00-.3-1.8l-.1-.1a2 2 0 112.8-2.8l.1.1a1.7 1.7 0 001.8.3H9a1.7 1.7 0 001-1.5V3a2 2 0 114 0v.1a1.7 1.7 0 001 1.5 1.7 1.7 0 001.8-.3l.1-.1a2 2 0 112.8 2.8l-.1.1a1.7 1.7 0 00-.3 1.8V9a1.7 1.7 0 001.5 1H21a2 2 0 110 4h-.1a1.7 1.7 0 00-1.5 1z",
  open: "M4 20h16M12 4v11M7 10l5 5 5-5",
  upload: "M4 20h16M12 15V4M7 9l5-5 5 5",
  save: "M12 4v11M7 10l5 5 5-5M4 20h16",
  play: "M7 5l12 7-12 7z",
  layers: "M12 3l9 5-9 5-9-5 9-5zM3 13l9 5 9-5",
  history: "M3 12a9 9 0 1 0 3-6.7L3 8M3 3v5h5M12 7v5l3 3",
  trash: "M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3",
  info: "M12 11v5M12 8v.01M12 21a9 9 0 100-18 9 9 0 000 18z",
  spark: "M12 3v4M12 17v4M3 12h4M17 12h4M6 6l2.5 2.5M15.5 15.5L18 18M6 18l2.5-2.5M15.5 8.5L18 6",
  file: "M14 3H7a2 2 0 00-2 2v14a2 2 0 002 2h10a2 2 0 002-2V8zM14 3v5h5",
  film: "M4 5h16v14H4zM8 5v14M16 5v14M4 9h4M4 15h4M16 9h4M16 15h4",
  "chevron-down": "M6 9l6 6 6-6",
  "chevron-up": "M6 15l6-6 6 6",
  stop: "M7 7h10v10H7z",
  "mark-in": "M8 5v14M8 12h10M14 8l4 4-4 4",
  "mark-out": "M16 5v14M16 12H6M10 8l-4 4 4 4",
  external: "M14 4h6v6M20 4l-9 9M18 14v5a1 1 0 01-1 1H5a1 1 0 01-1-1V7a1 1 0 011-1h5",
  compare: "M8 4v16M16 4v16M4 8h4M16 16h4M4 16h4M16 8h4",
  search: "M11 18a7 7 0 100-14 7 7 0 000 14zM20 20l-4-4",
  diff: "M6 3v12M6 15a3 3 0 100 6 3 3 0 000-6zM18 21V9M18 9a3 3 0 100-6 3 3 0 000 6zM6 9l12 0",
};

export function Icon({ name, className, size = 16 }: { name: string; className?: string; size?: number }) {
  return (
    <svg className={`icon${className ? ` ${className}` : ""}`} width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" focusable="false">
      <path d={PATHS[name] ?? PATHS.info} />
    </svg>
  );
}
