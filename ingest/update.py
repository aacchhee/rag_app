"""
Pull the notes repositories (ingest/repos.py) and ingest when something changed.

    python -m ingest.update     # what the 10-minute timer runs

Ingests when a repo got new commits, or when the repo list changed on the
ingest page (pending_ingest). A failed ingest is retried at the next check,
but at most once an hour, so a lasting problem doesn't fill the history.
The ingest page's "Start ingest" uses pull_and_ingest() too, so it always
works on the latest notes.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone

from ingest import repos, runs

_RETRY_AFTER_FAILURE = timedelta(hours=1)


def pull_and_ingest(trigger: str, lock, force_full: bool = False, only_if_changed: bool = False) -> bool:
    """
    With the ingest lock held: pull every repo, then ingest (always, or only if
    something changed). The lock is released when this returns. True if it ingested.
    """
    from ingest.ingest import main as run_ingest  # heavy imports: only when actually ingesting

    try:
        changed = repos.check_all()
        pending, failed_at = repos.pending_ingest()
        if only_if_changed:
            recently_failed = failed_at and datetime.now(timezone.utc) - datetime.fromisoformat(failed_at) < _RETRY_AFTER_FAILURE
            if not changed and (not pending or recently_failed):
                print("[update] no new commits; nothing to ingest", flush=True)
                runs.release_lock(lock)
                return False
        # Cleared before the run, under the lock: a repo change on the page while this
        # runs is refused (it needs the lock), so nothing set after this gets lost
        repos.set_pending(False)
    except BaseException:
        runs.release_lock(lock)
        raise

    try:
        run_ingest(force_full=force_full, trigger=trigger, lock=lock)  # releases the lock
    except BaseException:
        repos.set_pending(True, failed=True)
        raise
    return True


if __name__ == "__main__":
    if runs.is_running():
        print("[update] an ingest run is in progress; waiting for it to finish", flush=True)
    try:
        pull_and_ingest("timer", runs.acquire_lock(wait=True), only_if_changed=True)
    except repos.RepoError as e:
        sys.exit(f"[update] {e}")
