"""
Shared record of ingest runs, for every way a run can start: the ingest page
(ingest/web.py) and the command line (`python -m ingest.ingest`, which the
notes-repo trigger runs). All of them go through tracked_run(), so:

- only one run at a time: a file lock, held for the whole run, across processes
- the current or last run's state lives in a file, so it survives restarts
  and a command-line run shows on the status page too
- every run lands in the history: run_start, then run_done / run_error, or
  run_interrupted when the process died before it could say either
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import sys
import threading
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    import fcntl  # Linux, where the ingest runs
except ImportError:  # Windows dev machine: no cross-process lock
    fcntl = None

DATA_DIR = Path("/app/data")
STATE_FILE = DATA_DIR / "ingest_state.json"
HISTORY_FILE = DATA_DIR / "ingest_history.jsonl"
LOCK_FILE = DATA_DIR / "ingest.lock"

HISTORY_RETENTION_DAYS = 90
_MAX_LOG_CHARS = 100_000  # the state file keeps the end of the log
_SAVE_EVERY_SEC = 2.0  # how often a running job's log is written to the state file

TRIGGER_LABELS = {"page": "Ingest page", "cli": "Command line / notes update"}


class RunBusy(Exception):
    """Another ingest run holds the lock."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --- history -------------------------------------------------------------

def log_event(event_type: str, **extra) -> None:
    entry = {"t": _now(), "type": event_type, **extra}
    try:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(HISTORY_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass


def read_events(since: datetime | None = None) -> list[dict]:
    """History events, oldest first, within the retention period."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=HISTORY_RETENTION_DAYS)
    if since is not None and since > cutoff:
        cutoff = since
    events = []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    ev = json.loads(line)
                    if _parse(ev["t"]) >= cutoff:
                        events.append(ev)
                except (ValueError, KeyError, TypeError):
                    continue
    except OSError:
        pass
    events.sort(key=lambda ev: _parse(ev["t"]))  # appended in order anyway; a hand-edited file may not be
    return events


def list_runs(since: datetime | None = None) -> list[dict]:
    """
    Runs rebuilt from the history, newest first: each run_start paired with the
    run_done / run_error / run_interrupted that ended it. A start that was
    followed by another start without an end (history from before runs were
    tracked properly) counts as interrupted, its end unknown.
    """
    runs: list[dict] = []
    current: dict | None = None
    for ev in read_events(since):
        kind = ev.get("type")
        if kind == "run_start":
            if current is not None:
                current["result"] = "interrupted"
            current = {
                "started_at": ev["t"], "finished_at": None, "result": "running",
                "trigger": ev.get("trigger", "page"), "force_full": bool(ev.get("force_full")),
                "error": None, "duration_s": None,
            }
            runs.append(current)
        elif kind in ("run_done", "run_error", "run_interrupted") and current is not None:
            current["result"] = {"run_done": "done", "run_error": "error"}.get(kind, "interrupted")
            if kind != "run_interrupted":
                current["finished_at"] = ev["t"]
                current["duration_s"] = ev.get("duration_s") or round(
                    (_parse(ev["t"]) - _parse(current["started_at"])).total_seconds())
            current["error"] = ev.get("error")
            current = None
    # The last start may still be running, or may have died with no event yet
    if current is not None and read_state()["status"] != "running":
        current["result"] = "interrupted"
    return list(reversed(runs))


def _parse(ts: str) -> datetime:
    t = datetime.fromisoformat(ts)
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


# --- lock ----------------------------------------------------------------

def acquire_lock(wait: bool = False):
    """
    Hold the ingest lock (an open file). If another run has it: wait for it to
    finish when `wait`, else raise RunBusy.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fd = open(LOCK_FILE, "a+")
    if fcntl is not None:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
        except BlockingIOError:
            fd.close()
            raise RunBusy("an ingest run is already in progress")
    return fd


def release_lock(fd) -> None:
    with contextlib.suppress(OSError):
        if fcntl is not None:
            fcntl.flock(fd, fcntl.LOCK_UN)
        fd.close()


def is_running() -> bool:
    """Whether any process (this one included) holds the ingest lock."""
    try:
        fd = acquire_lock()
    except RunBusy:
        return True
    except OSError:
        return False
    release_lock(fd)
    return False


# --- state ---------------------------------------------------------------

_IDLE = {"status": "idle", "started_at": None, "finished_at": None, "error": None, "log": "",
         "trigger": None, "force_full": False}


def _write_state(state: dict) -> None:
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        tmp = STATE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, STATE_FILE)
    except OSError:
        pass


def read_state() -> dict:
    """
    The current run, or the last one: status is running, done, error or
    interrupted (idle: no run recorded yet). A run the file still calls
    running, while nobody holds the lock, died without finishing.
    """
    try:
        state = {**_IDLE, **json.loads(STATE_FILE.read_text(encoding="utf-8"))}
    except (OSError, ValueError):
        return dict(_IDLE)
    if state["status"] == "running" and not is_running():
        state.update(status="interrupted",
                     error="The run stopped without finishing: the service was restarted or the process crashed.")
        _write_state(state)
        log_event("run_interrupted", started_at=state["started_at"])
    return state


# --- tracking a run ------------------------------------------------------

_CR_OVERWRITE = re.compile(r"[^\n\r]*\r(?=[^\n])")  # progress bars redraw their line with \r


class _Tee(io.TextIOBase):
    """Passes output through to the real stream and keeps a copy for the state file."""

    def __init__(self, target, run: "_Run"):
        self.target, self.run = target, run

    def write(self, s):
        with contextlib.suppress(Exception):
            self.target.write(s)
        self.run.append(s)
        return len(s)

    def flush(self):
        with contextlib.suppress(Exception):
            self.target.flush()


class _Run:
    """The running job's state. A background thread writes new output to the
    state file every few seconds, so a killed run loses at most that much log."""

    def __init__(self, state: dict):
        self.state = state
        self._lock = threading.Lock()
        self._dirty = False
        self._stop = threading.Event()
        self._saver = threading.Thread(target=self._save_loop, daemon=True)

    def start(self) -> None:
        self.save()
        self._saver.start()

    def stop(self) -> None:
        self._stop.set()
        self._saver.join(timeout=5)

    def append(self, s: str) -> None:
        with self._lock:
            log = self.state["log"] + s
            if "\r" in s:
                head, tail = log[:-10_000], log[-10_000:]
                log = head + _CR_OVERWRITE.sub("", tail)
            if len(log) > _MAX_LOG_CHARS:
                log = "…(earlier output trimmed)…\n" + log[-_MAX_LOG_CHARS:]
            self.state["log"] = log
            self._dirty = True

    def update(self, **fields) -> None:
        with self._lock:
            self.state.update(fields)
            self._dirty = True

    def save(self) -> None:
        with self._lock:
            snapshot = dict(self.state)
            self._dirty = False
        _write_state(snapshot)

    def _save_loop(self) -> None:
        while not self._stop.wait(_SAVE_EVERY_SEC):
            if self._dirty:
                self.save()


@contextlib.contextmanager
def tracked_run(trigger: str, force_full: bool = False, lock=None):
    """
    Run the body as a recorded ingest: hold the lock, keep state and log in the
    state file, write the history. Exceptions are recorded and re-raised.
    `lock`: one acquire_lock() already returned (the page takes it before
    answering, so two clicks can't both start a run).
    """
    lock = lock if lock is not None else acquire_lock()  # RunBusy: nothing recorded
    started = time.monotonic()
    run = _Run({**_IDLE, "status": "running", "started_at": _now(), "trigger": trigger,
                "force_full": force_full})
    try:
        run.start()
        log_event("run_start", trigger=trigger, force_full=force_full)
        with contextlib.redirect_stdout(_Tee(sys.stdout, run)), contextlib.redirect_stderr(_Tee(sys.stderr, run)):
            yield
    except BaseException as e:
        summary = (traceback.format_exception_only(type(e), e)[-1].strip())[:300]
        run.update(status="error", error=traceback.format_exc())
        log_event("run_error", duration_s=round(time.monotonic() - started), error=summary)
        raise
    else:
        run.update(status="done")
        log_event("run_done", duration_s=round(time.monotonic() - started))
    finally:
        run.stop()
        run.update(finished_at=_now())
        run.save()
        release_lock(lock)
