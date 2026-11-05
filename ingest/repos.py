"""
The notes repositories the ingest reads, managed from the ingest page instead
of being listed in code. Stored in /app/data/repos.json:

    {"repos": [{"course": "TMA4411", "url": "https://github.com/owner/name",
                "path": "/app/data/repos/TMA4411", "site_url": "https://owner.github.io/name/",
                "exclude": ["*fasit*.pdf"], "last_check": {...}}],
     "pending_ingest": false, "failed_at": null}

Each repo is a clone on the server, kept identical to its GitHub default
branch (fetch + hard reset: nobody edits these clones). The ingest page adds,
edits and removes repos; `python -m ingest.update` (the 10-minute timer) pulls
them and ingests when something changed.

Every change here runs under the ingest lock (ingest/runs.py), so a clone is
never pulled or removed while an ingest reads it.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ingest.runs import DATA_DIR

REPOS_FILE = DATA_DIR / "repos.json"
CLONES_DIR = DATA_DIR / "repos"

# Only used once, to create repos.json from the repos that used to be listed in code
_LEGACY = [
    {"course": "IMAx2024", "path": "/home/rag/notes_repo", "site_url": "https://aacchhee.github.io/ingmat2_ntnu/"},
    {"course": "IMAx3011", "path": "/home/rag/ingmat3a_ntnu", "site_url": "https://aacchhee.github.io/ingmat3a_ntnu/"},
]

_GIT = shutil.which("git") or "/usr/bin/git"  # the service's PATH only has its venv
_GIT_TIMEOUT = 300
_GITHUB_RE = re.compile(r"^(?:https://github\.com/|git@github\.com:)([A-Za-z0-9-]+)/([A-Za-z0-9._-]+?)(?:\.git)?/?$")
_COURSE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class RepoError(Exception):
    """A repo operation failed; the message is meant for the person on the ingest page."""


@dataclass
class Repo:
    course: str
    url: str
    path: Path
    site_url: str | None = None
    exclude: list[str] = field(default_factory=list)
    last_check: dict = field(default_factory=dict)

    def excluded(self, rel: str) -> bool:
        """Whether a file (path inside the repo, with /) matches one of the skip patterns."""
        name = rel.rsplit("/", 1)[-1]
        return any(fnmatch.fnmatch(rel, p) or fnmatch.fnmatch(name, p) for p in self.exclude)

    def to_json(self) -> dict:
        return {"course": self.course, "url": self.url, "path": str(self.path), "site_url": self.site_url,
                "exclude": self.exclude, "last_check": self.last_check}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --- the settings file -----------------------------------------------------

def _load() -> dict:
    try:
        data = json.loads(REPOS_FILE.read_text(encoding="utf-8"))
        data.setdefault("repos", [])
        return data
    except FileNotFoundError:
        return _seed()
    except (OSError, ValueError) as e:
        raise RepoError(f"could not read {REPOS_FILE}: {e}") from e


def _save(data: dict) -> None:
    REPOS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = REPOS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, REPOS_FILE)


def _seed() -> dict:
    """First start without repos.json: take over the clones the old code listed."""
    repos = []
    for old in _LEGACY:
        path = Path(old["path"])
        if not (path / ".git").exists():
            continue
        try:
            url = canonical_url(_git(path, "remote", "get-url", "origin"))
        except (RepoError, ValueError):
            continue
        repos.append(Repo(old["course"], url, path, old["site_url"]).to_json())
    data = {"repos": repos, "pending_ingest": False, "failed_at": None}
    if repos:
        try:
            _save(data)
        except OSError:
            pass  # e.g. a dev machine without /app/data: just use it in memory
    return data


def list_repos() -> list[Repo]:
    try:
        entries = _load()["repos"]
    except RepoError:
        return []
    return [Repo(e["course"], e["url"], Path(e["path"]), e.get("site_url"), list(e.get("exclude") or []),
                 dict(e.get("last_check") or {})) for e in entries]


def repo_courses() -> set[str]:
    return {r.course for r in list_repos()}


def pending_ingest() -> tuple[bool, str | None]:
    data = _load()
    return bool(data.get("pending_ingest")), data.get("failed_at")


def set_pending(pending: bool, failed: bool = False) -> None:
    data = _load()
    data["pending_ingest"] = pending
    data["failed_at"] = _now() if failed else None
    _save(data)


def _update_entry(course: str, **fields) -> None:
    data = _load()
    for e in data["repos"]:
        if e["course"] == course:
            e.update(fields)
    _save(data)


# --- URLs ------------------------------------------------------------------

def canonical_url(url: str) -> str:
    """https://github.com/owner/name for any GitHub repo URL; ValueError otherwise."""
    m = _GITHUB_RE.match(url.strip())
    if not m:
        raise ValueError("not a GitHub repository link (expected https://github.com/owner/repo)")
    return f"https://github.com/{m.group(1)}/{m.group(2)}"


def default_site_url(url: str) -> str:
    """Where GitHub Pages publishes the repo: https://owner.github.io/name/."""
    owner, name = canonical_url(url).split("/")[-2:]
    owner = owner.lower()
    if name.lower() == f"{owner}.github.io":
        return f"https://{owner}.github.io/"
    return f"https://{owner}.github.io/{name}/"


# --- git -------------------------------------------------------------------

def _git(cwd: Path | None, *args: str, timeout: int = _GIT_TIMEOUT) -> str:
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "LC_ALL": "C"}  # a private repo fails instead of asking
    try:
        p = subprocess.run([_GIT, *args], cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise RepoError(f"git {args[0]} took longer than {timeout} s") from e
    except OSError as e:
        raise RepoError(f"could not run git: {e}") from e
    if p.returncode != 0:
        msg = (p.stderr or p.stdout).strip().splitlines()
        hint = " (is the repository public?)" if any("could not read Username" in m or "not found" in m
                                                    for m in msg) else ""
        raise RepoError(f"git {args[0]} failed: {msg[-1] if msg else 'exit ' + str(p.returncode)}{hint}")
    return p.stdout.strip()


def _head(path: Path) -> dict:
    sha, subject, date = _git(path, "log", "-1", "--format=%h%x00%s%x00%cI").split("\0")
    return {"commit": sha, "subject": subject[:120], "date": date}


def _sync(path: Path) -> bool:
    """Make the clone identical to its remote default branch. True if that changed it."""
    before = _git(path, "rev-parse", "HEAD")
    _git(path, "fetch", "--prune", "origin")
    _git(path, "remote", "set-head", "origin", "--auto")  # the default branch, in case it changed
    target = _git(path, "rev-parse", "origin/HEAD")
    if target != before:
        _git(path, "reset", "--hard", "origin/HEAD")
    _git(path, "clean", "-fd")
    return target != before


def check(repo: Repo) -> bool:
    """Pull one repo and record the result. True if it got new commits."""
    try:
        changed = _sync(repo.path)
        info = {"at": _now(), **_head(repo.path), "error": None}
    except RepoError as e:
        changed, info = False, {**repo.last_check, "at": _now(), "error": str(e)}
    _update_entry(repo.course, last_check=info)
    return changed


def check_all() -> bool:
    """Pull every repo (lock held by the caller). True if any got new commits."""
    changed = False
    for repo in list_repos():
        if check(repo):
            print(f"[repos] {repo.course}: new commits in {repo.url}", flush=True)
            changed = True
    if changed:
        set_pending(True)
    return changed


# --- managing the list (from the ingest page, lock held by the caller) ------

def _check_course(course: str) -> str:
    course = (course or "").strip()
    if not _COURSE_RE.match(course):
        raise RepoError("course name: letters, digits, '-', '_' or '.', starting with a letter or digit")
    return course


def _clean_patterns(exclude) -> list[str]:
    if isinstance(exclude, str):
        exclude = re.split(r"[,\n]", exclude)
    return [p.strip() for p in (exclude or []) if p and p.strip()]


def add_repo(url: str, course: str, site_url: str | None = None, exclude=None) -> Repo:
    try:
        url = canonical_url(url)
    except ValueError as e:
        raise RepoError(str(e)) from e
    course = _check_course(course)
    existing = list_repos()
    if any(r.course == course for r in existing):
        raise RepoError(f"course {course} already has a repository; edit or remove that one")
    if any(r.url.lower() == url.lower() for r in existing):
        raise RepoError("this repository is already added")

    dest = CLONES_DIR / course
    if dest.exists():
        shutil.rmtree(dest)  # leftover from a failed or removed clone
    CLONES_DIR.mkdir(parents=True, exist_ok=True)
    try:
        _git(None, "clone", "--quiet", url + ".git", str(dest))
        head = _head(dest)
    except RepoError:
        shutil.rmtree(dest, ignore_errors=True)
        raise

    repo = Repo(course, url, dest, (site_url or "").strip() or default_site_url(url), _clean_patterns(exclude),
                {"at": _now(), **head, "error": None})
    data = _load()
    data["repos"].append(repo.to_json())
    data["pending_ingest"] = True
    _save(data)
    return repo


def update_repo(course: str, url: str | None = None, site_url: str | None = None, exclude=None) -> Repo:
    repo = next((r for r in list_repos() if r.course == course), None)
    if repo is None:
        raise RepoError(f"no repository for course {course}")
    reingest = False
    if url is not None:
        try:
            url = canonical_url(url)
        except ValueError as e:
            raise RepoError(str(e)) from e
        if url.lower() != repo.url.lower():
            # Same clone, new address: files that stay the same are not re-embedded
            old = repo.url
            _git(repo.path, "remote", "set-url", "origin", url + ".git")
            try:
                _sync(repo.path)
            except RepoError:
                _git(repo.path, "remote", "set-url", "origin", old + ".git")
                raise
            repo.url, reingest = url, True
            repo.last_check = {"at": _now(), **_head(repo.path), "error": None}
    if site_url is not None:
        repo.site_url = site_url.strip() or default_site_url(repo.url)
    if exclude is not None:
        patterns = _clean_patterns(exclude)
        reingest = reingest or patterns != repo.exclude
        repo.exclude = patterns
    data = _load()
    data["repos"] = [repo.to_json() if e["course"] == course else e for e in data["repos"]]
    if reingest:
        data["pending_ingest"] = True
    _save(data)
    return repo


def remove_repo(course: str) -> None:
    data = _load()
    entry = next((e for e in data["repos"] if e["course"] == course), None)
    if entry is None:
        raise RepoError(f"no repository for course {course}")
    data["repos"] = [e for e in data["repos"] if e["course"] != course]
    data["pending_ingest"] = True  # the next ingest removes its files from the index
    _save(data)
    path = Path(entry["path"])
    # Only delete clones this module made; an old clone elsewhere is left for a person to remove
    if CLONES_DIR in path.parents:
        shutil.rmtree(path, ignore_errors=True)
