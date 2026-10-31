// Minimal accessible modal dialog on the native <dialog> element.
import { useEffect, useRef, type ReactNode } from "react";

export function Dialog({ open, title, children, actions, onClose }: { open: boolean; title: string; children: ReactNode; actions: ReactNode; onClose: () => void }) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (open && !el.open) {
      if (typeof el.showModal === "function") el.showModal();
      else el.setAttribute("open", "");
    } else if (!open && el.open) {
      if (typeof el.close === "function") el.close();
      else el.removeAttribute("open");
    }
  }, [open]);
  return (
    <dialog ref={ref} className="dialog" aria-labelledby="dialog-title" onCancel={(e) => (e.preventDefault(), onClose())} onClick={(e) => e.target === e.currentTarget && onClose()}>
      {open ? (
        <div className="dialog__box">
          <h2 id="dialog-title">{title}</h2>
          <div className="dialog__body">{children}</div>
          <div className="dialog__actions">{actions}</div>
        </div>
      ) : null}
    </dialog>
  );
}
