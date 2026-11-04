// The preset list of the server (built-in + user), loaded on mount and
// refreshed after a save or a delete.
import { useCallback, useEffect, useRef, useState } from "react";
import type { PresetInfo, PresetsApi } from "../api/presets";

export function usePresets(api: PresetsApi) {
  const [presets, setPresets] = useState<PresetInfo[]>([]);
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");
  const [error, setError] = useState<string | null>(null);
  const seq = useRef(0);

  const refresh = useCallback(async () => {
    const mine = ++seq.current; // an older answer never overwrites a newer one
    try {
      const list = await api.list();
      if (mine !== seq.current) return;
      setPresets(list);
      setError(null);
      setStatus("ready");
    } catch (exc) {
      if (mine !== seq.current) return;
      setError(exc instanceof Error ? exc.message : String(exc));
      setStatus("error");
    }
  }, [api]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  return { presets, status, error, refresh };
}
