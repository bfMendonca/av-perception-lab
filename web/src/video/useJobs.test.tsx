import { act, renderHook, waitFor } from "@testing-library/react";
import { ApiError } from "../api/client";
import type { JobEvent, JobInfo, VideoApi } from "../api/video";
import { LOST_ERROR, useJobs } from "./useJobs";

function info(id: string, patch: Partial<JobInfo> = {}): JobInfo {
  return { id, status: "running", position: null, error: null, run_id: null, frames: 0, log_tail: [], video: "road.mp4", ...patch };
}

function fakeApi(listed: JobInfo[], known: Map<string, JobInfo>) {
  const listeners = new Map<string, { onEvent: (e: JobEvent) => void; onError?: () => void }>();
  const api = {
    listJobs: async () => listed,
    getJob: async (id: string) => {
      const found = known.get(id);
      if (!found) throw new ApiError(`unknown job '${id}'`, 404);
      return found;
    },
    jobEvents: (id: string, onEvent: (e: JobEvent) => void, onError?: () => void) => {
      listeners.set(id, { onEvent, onError });
      return () => listeners.delete(id);
    },
  } as unknown as VideoApi;
  return { api, listeners };
}

describe("useJobs", () => {
  it("follows the server's unfinished jobs after a reload", async () => {
    const running = info("j1", { frames: 7, run_id: "r1" });
    const done = info("j0", { status: "done" });
    const { api, listeners } = fakeApi([running, done], new Map([["j1", running]]));
    const { result } = renderHook(() => useJobs(api));
    await waitFor(() => expect(result.current.jobs.map((j) => j.id)).toEqual(["j1"]));
    expect(result.current.jobs[0]).toMatchObject({ status: "running", frames: 7, label: "road.mp4", runId: "r1" });
    expect(listeners.has("j1")).toBe(true); // re-subscribed; finished jobs are not
    // A replay from the start never rewinds the progress nor the status.
    act(() => listeners.get("j1")?.onEvent({ event: "queued" }));
    act(() => listeners.get("j1")?.onEvent({ event: "frame", n: 1 }));
    expect(result.current.jobs[0]).toMatchObject({ status: "running", frames: 7 });
    act(() => listeners.get("j1")?.onEvent({ event: "frame", n: 9 }));
    expect(result.current.jobs[0].frames).toBe(9);
  });

  it("settles a job the restarted server no longer knows", async () => {
    const running = info("j2");
    const { api, listeners } = fakeApi([running], new Map());
    const { result } = renderHook(() => useJobs(api));
    await waitFor(() => expect(listeners.has("j2")).toBe(true));
    act(() => listeners.get("j2")?.onError?.());
    await waitFor(() => expect(result.current.jobs[0].status).toBe("failed"));
    expect(result.current.jobs[0].error).toBe(LOST_ERROR);
  });
});
