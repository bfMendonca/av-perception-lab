"""JobQueue: queueing order, cancel, crash, event replay (fake worker, no GPU)."""

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from det_depth_distance.web.jobs import JobQueue
from det_depth_distance.web.paths import WebPaths

FAKE = Path(__file__).with_name("fake_worker.py")


def argv(path: Path) -> list[str]:
    return [sys.executable, str(FAKE), str(path)]


def make_queue(tmp_path: Path) -> JobQueue:
    return JobQueue(WebPaths(tmp_path / "out", tmp_path / "lib"), argv)


async def wait_for(predicate, timeout=15.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        assert asyncio.get_running_loop().time() < deadline, "timed out"
        await asyncio.sleep(0.02)


async def collect(queue, job_id):
    return [e async for _, e in queue.events(job_id)]


def test_second_job_waits_for_first(tmp_path):
    mark = tmp_path / "mark.txt"

    async def scenario():
        queue = make_queue(tmp_path)
        try:
            jobs = [
                queue.submit(
                    {"fake": {"name": n, "mark": str(mark), "seconds": 0.4}}, None
                )
                for n in ("a", "b", "c")
            ]
            assert [queue.position(j) for j in jobs] == [0, 1, 2]
            assert jobs[1].status == "queued" and jobs[2].status == "queued"
            for j in jobs:
                await collect(queue, j.id)
            return [j.status for j in jobs]
        finally:
            await queue.aclose()

    assert asyncio.run(scenario()) == ["done"] * 3
    # Strictly one at a time and in submission order.
    assert mark.read_text().split("\n")[:-1] == [
        "start a",
        "end a",
        "start b",
        "end b",
        "start c",
        "end c",
    ]


def test_cancel_running_job_kills_worker_and_marks_cancelled(tmp_path):
    pid_file = tmp_path / "pid"

    async def scenario():
        queue = make_queue(tmp_path)
        try:
            job = queue.submit(
                {"fake": {"mode": "sleep", "name": "x", "pid_file": str(pid_file)}}
            )
            nxt = queue.submit({"fake": {"name": "y"}})
            await wait_for(lambda: job.frames == 1)
            queue.cancel(job.id)
            assert job.status == "cancelled"
            events = await collect(queue, job.id)
            assert events[-1] == {"event": "end", "status": "cancelled", "error": None}
            # The queue is free: the next job runs to completion.
            await collect(queue, nxt.id)
            assert nxt.status == "done"
            return job
        finally:
            await queue.aclose()

    job = asyncio.run(scenario())
    pid = int(pid_file.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    meta = json.loads((Path(job.run_dir) / "web_run.json").read_text())
    assert meta["status"] == "cancelled"  # history flags the run as incomplete


def test_cancel_queued_job_never_starts(tmp_path):
    mark = tmp_path / "mark.txt"

    async def scenario():
        queue = make_queue(tmp_path)
        try:
            first = queue.submit({"fake": {"name": "a", "mark": str(mark)}})
            second = queue.submit({"fake": {"name": "b", "mark": str(mark)}})
            queue.cancel(second.id)
            events = await collect(queue, second.id)
            await collect(queue, first.id)
            return second, events
        finally:
            await queue.aclose()

    second, events = asyncio.run(scenario())
    assert second.status == "cancelled" and events[-1]["event"] == "end"
    assert "start b" not in mark.read_text()


def test_worker_crash_marks_failed_with_log_tail(tmp_path):
    async def scenario():
        queue = make_queue(tmp_path)
        try:
            job = queue.submit({"fake": {"mode": "crash", "name": "z"}})
            after = queue.submit({"fake": {"name": "w"}})
            events = await collect(queue, job.id)
            await collect(queue, after.id)
            return job, after, events
        finally:
            await queue.aclose()

    job, after, events = asyncio.run(scenario())
    assert job.status == "failed"
    assert "CUDA out of memory" in job.log_tail[-1]
    assert "code 1" in job.error and "CUDA out of memory" in job.error
    assert events[-1]["status"] == "failed"
    assert after.status == "done"  # the crash freed the queue
    meta = json.loads((Path(job.run_dir) / "web_run.json").read_text())
    assert meta["status"] == "failed"


def test_events_stream_ends_after_done(tmp_path):
    async def scenario():
        queue = make_queue(tmp_path)
        try:
            job = queue.submit({"fake": {"name": "e"}})
            live = await collect(queue, job.id)
            late = await collect(queue, job.id)  # replays from the start
            return live, late
        finally:
            await queue.aclose()

    live, late = asyncio.run(scenario())
    kinds = [e["event"] for e in live]
    assert kinds[0] == "queued"
    assert kinds[-2:] == ["done", "end"]
    assert "frame" in kinds and "started" in kinds
    assert late == live


def test_submit_forces_out_root_and_rejects_bad_video_id(tmp_path):
    async def scenario():
        queue = make_queue(tmp_path)
        try:
            job = queue.submit({"output": {"out_root": "/etc", "tag": "t"}})
            for bad in ("../x", "a/b", "/abs", ".."):
                with pytest.raises(ValueError):
                    queue.submit({}, bad)
            await collect(queue, job.id)
            return job
        finally:
            await queue.aclose()

    job = asyncio.run(scenario())
    assert job.config["output"] == {
        "out_root": str(tmp_path / "out"),
        "tag": f"web-{job.id}",
    }


def test_done_event_sets_run_dir_and_job_record_is_persisted(tmp_path):
    async def scenario():
        queue = make_queue(tmp_path)
        try:
            job = queue.submit({"fake": {"mode": "wrongstart", "name": "d"}})
            await collect(queue, job.id)
            return job
        finally:
            await queue.aclose()

    job = asyncio.run(scenario())
    assert job.status == "done"
    assert job.run_dir.endswith(f"_d_web-{job.id}")  # taken from `done`
    record = json.loads((tmp_path / "out/.web_jobs" / f"{job.id}.json").read_text())
    assert record["status"] == "done" and record["run_dir"] == job.run_dir


def test_stderr_without_newline_does_not_break_done(tmp_path):
    async def scenario():
        queue = make_queue(tmp_path)
        try:
            job = queue.submit({"fake": {"mode": "partial_stderr", "name": "p"}})
            await collect(queue, job.id)
            return job
        finally:
            await queue.aclose()

    job = asyncio.run(scenario())
    assert job.status == "done" and job.frames == 2
    assert "downloading 50%" in job.log_tail


def test_failure_before_started_leaves_a_failed_history_entry(tmp_path):
    from det_depth_distance.web.history import load_run, scan_runs

    async def scenario():
        queue = make_queue(tmp_path)
        try:
            job = queue.submit({"fake": {"mode": "crash_early"}})
            await collect(queue, job.id)
            return job
        finally:
            await queue.aclose()

    job = asyncio.run(scenario())
    assert job.status == "failed" and job.run_dir is None
    (rec,) = scan_runs(tmp_path / "out")  # `.web_jobs` itself is not a run
    assert rec.id == f"job-{job.id}" and rec.status == "failed"
    assert rec.complete is False and "CUDA out of memory (load)" in rec.error
    assert load_run(tmp_path / "out", rec.id) == rec


def test_cancel_before_started_leaves_a_cancelled_history_entry(tmp_path):
    from det_depth_distance.web.history import scan_runs

    async def scenario():
        queue = make_queue(tmp_path)
        try:
            job = queue.submit({"fake": {"mode": "sleep_early"}})
            await wait_for(lambda: job.status == "running")
            queue.cancel(job.id)
            await collect(queue, job.id)
            return job
        finally:
            await queue.aclose()

    job = asyncio.run(scenario())
    (rec,) = scan_runs(tmp_path / "out")
    assert (rec.id, rec.status, rec.complete) == (f"job-{job.id}", "cancelled", False)


def test_aclose_ends_running_and_queued_jobs(tmp_path):
    # Server shutdown: no record (nor run folder) stays running/queued.
    pid_file = tmp_path / "pid"

    async def scenario():
        queue = make_queue(tmp_path)
        running = queue.submit(
            {"fake": {"mode": "sleep", "name": "x", "pid_file": str(pid_file)}}
        )
        queued = queue.submit({"fake": {"name": "y"}})
        await wait_for(lambda: running.frames == 1)
        await queue.aclose()
        events = await collect(queue, running.id)  # subscribers are woken
        return running, queued, events

    running, queued, events = asyncio.run(scenario())
    assert events[-1] == {
        "event": "end",
        "status": "cancelled",
        "error": "the server shut down",
    }
    records = tmp_path / "out" / ".web_jobs"
    for job in (running, queued):
        record = json.loads((records / f"{job.id}.json").read_text())
        assert record["status"] == "cancelled"
        assert record["error"] == "the server shut down"
    meta = json.loads((Path(running.run_dir) / "web_run.json").read_text())
    assert meta["status"] == "cancelled"
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)


def test_leftover_unfinished_records_are_settled_on_start(tmp_path):
    # A server killed before aclose ran (kill -9, power loss).
    records = tmp_path / "out" / ".web_jobs"
    records.mkdir(parents=True)
    run_dir = tmp_path / "out" / "20260101_000000_x_web-r1"
    run_dir.mkdir()
    (run_dir / "web_run.json").write_text(json.dumps({"status": "running"}))
    rows = {
        "r1": {"id": "r1", "status": "running", "run_dir": str(run_dir)},
        "q1": {"id": "q1", "status": "queued"},
        "d1": {"id": "d1", "status": "done"},
    }
    for job_id, row in rows.items():
        (records / f"{job_id}.json").write_text(json.dumps(row))
    make_queue(tmp_path)
    read = {k: json.loads((records / f"{k}.json").read_text()) for k in rows}
    assert read["r1"]["status"] == "failed" and "interrupted" in read["r1"]["error"]
    assert read["q1"]["status"] == "cancelled" and "interrupted" in read["q1"]["error"]
    assert read["d1"] == rows["d1"]
    meta = json.loads((run_dir / "web_run.json").read_text())
    assert meta["status"] == "failed"


def test_recent_lists_unfinished_and_latest_finished(tmp_path, monkeypatch):
    from det_depth_distance.web import jobs as jobs_module

    monkeypatch.setattr(jobs_module, "RECENT_FINISHED", 1)

    async def scenario():
        queue = make_queue(tmp_path)
        try:
            a = queue.submit({"fake": {"name": "a"}})
            b = queue.submit({"fake": {"name": "b"}})
            await collect(queue, a.id)
            await collect(queue, b.id)
            c = queue.submit({"fake": {"mode": "sleep", "name": "c"}})
            listed = [j.id for j in queue.recent()]
            queue.cancel(c.id)
            return listed, (a.id, b.id, c.id)
        finally:
            await queue.aclose()

    listed, (_a, b, c) = asyncio.run(scenario())
    assert listed == [c, b]  # newest first; only the latest finished one


def test_events_resume_after_an_index(tmp_path):
    async def scenario():
        queue = make_queue(tmp_path)
        try:
            job = queue.submit({"fake": {"name": "a"}})
            every = [i for i, _ in [x async for x in queue.events(job.id)]]
            later = [x async for x in queue.events(job.id, 3)]
            return every, later, job
        finally:
            await queue.aclose()

    every, later, job = asyncio.run(scenario())
    assert every == list(range(len(job.events)))
    assert [i for i, _ in later] == every[3:]
    assert [e for _, e in later] == job.events[3:]
