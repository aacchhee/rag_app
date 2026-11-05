"""
Minimal web UI for the ingest service.

Wraps ingest.ingest.main() so a re-index can be triggered from a browser
instead of a shell, with live status/log output. Runs as a small Flask app;
the actual ingest logic in ingest/ingest.py is untouched.
"""

import os
import signal
import threading
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, Response, jsonify, request
from werkzeug.utils import secure_filename

from ingest import repos, runs, update
from ingest.ingest import (
    _load_manifest, _rel_source, COURSES_DIR, list_courses, load_copyrighted,
    save_copyrighted, set_copyrighted_in_qdrant,
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB per upload

# Exit after 30 minutes without requests, unless a run is in progress: frees the
# memory marker's models take. Run state and history live in files (ingest/runs.py),
# so they survive the restart.
_last_request_time = datetime.now(timezone.utc)
_AUTO_EXIT_DELAY_SEC = int(os.getenv("INGEST_AUTO_EXIT_DELAY_MIN", "30")) * 60


def _bump_activity():
    global _last_request_time
    _last_request_time = datetime.now(timezone.utc)


def _schedule_auto_exit():
    def _watchdog():
        while True:
            threading.Event().wait(timeout=_AUTO_EXIT_DELAY_SEC)
            idle_for = (datetime.now(timezone.utc) - _last_request_time).total_seconds()
            if idle_for >= _AUTO_EXIT_DELAY_SEC and not runs.is_running():
                os.kill(os.getpid(), signal.SIGTERM)
    threading.Thread(target=_watchdog, daemon=True).start()


_schedule_auto_exit()


def _run_job(lock, force_full: bool):
    try:
        # Pull the notes repositories first, so the run works on the latest notes
        update.pull_and_ingest("page", lock, force_full=force_full)
    except Exception:
        pass  # recorded by runs.tracked_run: status page and history show it


@app.get("/")
def index():
    _bump_activity()
    return Response(_PAGE, mimetype="text/html")


@app.get("/history")
def get_history():
    # Not counted as activity: the public status page polls this
    events = runs.read_events()
    return jsonify(
        runs=runs.list_runs(),  # newest first: started/finished, result, trigger, duration
        first_event=events[0]["t"] if events else None,  # before this, the chart has no data
        trigger_labels=runs.TRIGGER_LABELS,
        events=list(reversed(events[-10000:])),
    )


@app.get("/sources")
def list_sources():
    _bump_activity()
    manifest = _load_manifest()
    if manifest is None:
        return jsonify({"indexed": False, "sources": []})
    sources = manifest.get("sources", {})
    course_filter = (request.args.get("course") or "").strip()

    items = [
        {
            # Repo files are keyed "COURSE:path" in the manifest; show the path, the course has its column
            "source": meta.get("source", key),
            "chunks": len(meta.get("chunk_ids", [])),
            "hash": meta.get("hash", "")[:16] + "…",
            "course": meta.get("course"),
        }
        for key, meta in sorted(sources.items())
    ]

    if course_filter:
        items = [item for item in items if item["course"] == course_filter]

    return jsonify(
        {
            "indexed": True,
            "collection": manifest.get("qdrant_collection"),
            "embedding_dim": manifest.get("embedding_dim"),
            "total_sources": len(items),
            "total_chunks": sum(i["chunks"] for i in items),
            "sources": items,
        }
    )


@app.get("/status")
def status():
    # The current run, or the last one (also one started from the command line)
    return jsonify(runs.read_state())


@app.post("/run")
def trigger():
    _bump_activity()
    force_full = bool((request.get_json(silent=True) or {}).get("force_full"))
    # Take the lock here, before answering: of two quick clicks only one starts a run
    try:
        lock = runs.acquire_lock()
    except runs.RunBusy:
        return jsonify({"error": "an ingest run is already in progress"}), 409
    threading.Thread(target=_run_job, args=(lock, force_full), daemon=True).start()
    return jsonify({"started": True, "force_full": force_full})


# --- Courses ---

@app.get("/admin/courses")
def get_courses():
    _bump_activity()
    courses = list_courses()
    return jsonify(courses)


@app.post("/admin/courses")
def create_course():
    _bump_activity()
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "course name required"}), 400
    safe = secure_filename(name)
    if safe != name or not safe:
        return jsonify({"error": "invalid course name"}), 400
    target = COURSES_DIR / safe
    target.mkdir(parents=True, exist_ok=True)
    return jsonify({"created": safe})


@app.delete("/admin/courses/<path:name>")
def delete_course(name):
    _bump_activity()
    if name in repos.repo_courses():
        return jsonify({"error": f"{name} has a notes repository: remove the repository first"}), 400
    safe = secure_filename(name)
    target = COURSES_DIR / safe
    if not safe or not target.is_dir():
        return jsonify({"error": f"{name} has no upload folder; its indexed files disappear "
                                 "when their source is removed and an ingest runs"}), 404
    # Only delete if empty
    files = list(target.rglob("*"))
    if any(p.is_file() for p in files):
        return jsonify({"error": "course is not empty"}), 400
    try:
        target.rmdir()
        return jsonify({"deleted": safe})
    except OSError:
        return jsonify({"error": "could not delete"}), 500


# --- Notes repositories ---

def _repos_json():
    pending, failed_at = repos.pending_ingest()
    return {"repos": [r.to_json() for r in repos.list_repos()], "pending_ingest": pending, "failed_at": failed_at}


def _locked(action):
    """Run a repo change with the ingest lock: never while an ingest reads the clones."""
    try:
        lock = runs.acquire_lock()
    except runs.RunBusy:
        return jsonify({"error": "an ingest is running; try again when it has finished"}), 409
    try:
        action()
        return jsonify(_repos_json())
    except repos.RepoError as e:
        return jsonify({"error": str(e)}), 400
    finally:
        runs.release_lock(lock)


@app.get("/admin/repos")
def get_repos():
    _bump_activity()
    return jsonify(_repos_json())


@app.post("/admin/repos")
def add_repo():
    _bump_activity()
    data = request.get_json(silent=True) or {}

    def action():
        repo = repos.add_repo(data.get("url", ""), data.get("course", ""), data.get("site_url"), data.get("exclude"))
        runs.log_event("repo_added", course=repo.course, url=repo.url)
    return _locked(action)


@app.patch("/admin/repos/<course>")
def edit_repo(course):
    _bump_activity()
    data = request.get_json(silent=True) or {}

    def action():
        repo = repos.update_repo(course, data.get("url"), data.get("site_url"), data.get("exclude"))
        runs.log_event("repo_changed", course=repo.course, url=repo.url)
    return _locked(action)


@app.delete("/admin/repos/<course>")
def remove_repo(course):
    _bump_activity()

    def action():
        repos.remove_repo(course)
        runs.log_event("repo_removed", course=course)
    return _locked(action)


# --- Uploaded files (course-scoped) ---

SUPPORTED_EXTS = {"pdf", "md", "qmd"}

def _course_upload_dir(course: str | None) -> Path:
    if course:
        return COURSES_DIR / secure_filename(course)
    return COURSES_DIR  # fallback


@app.get("/files")
def list_files():
    _bump_activity()
    course = (request.args.get("course") or "").strip()
    d = _course_upload_dir(course) if course else COURSES_DIR
    if not d.exists():
        return jsonify([])
    copyrighted = load_copyrighted()
    files = sorted(
        (p for p in d.rglob("*") if p.is_file() and p.suffix.lower().lstrip(".") in SUPPORTED_EXTS),
        key=lambda p: p.name,
    )
    return jsonify([
        {"name": p.name, "source": _rel_source(p), "copyrighted": _rel_source(p) in copyrighted}
        for p in files
    ])


@app.post("/files")
def upload_file():
    _bump_activity()
    files = request.files.getlist("file")
    files = [f for f in files if f and f.filename]
    if not files:
        return jsonify({"error": "no file provided"}), 400

    course = (request.form.get("course") or "").strip()
    target_dir = _course_upload_dir(course) if course else COURSES_DIR / "uncategorized"
    target_dir.mkdir(parents=True, exist_ok=True)

    uploaded: list[str] = []
    skipped: list[str] = []
    mark_copyrighted = request.form.get("copyrighted") == "true"
    copyrighted = load_copyrighted()

    for file in files:
        filename = secure_filename(file.filename)
        ext = Path(filename).suffix.lower().lstrip(".")
        if ext not in SUPPORTED_EXTS:
            skipped.append(f"{filename} (unsupported type)")
            continue

        if ext == "pdf":
            head = file.stream.read(5)
            file.stream.seek(0)
            if head != b"%PDF-":
                skipped.append(f"{filename} (does not look like a PDF)")
                continue

        file.save(target_dir / filename)
        uploaded.append(filename)
        if mark_copyrighted:
            copyrighted.add(_rel_source(target_dir / filename))

    if not uploaded and skipped:
        return jsonify({"error": "no valid files to upload", "skipped": skipped}), 400
    if mark_copyrighted:
        save_copyrighted(copyrighted)

    return jsonify({
        "uploaded": uploaded,
        "uploaded_count": len(uploaded),
        "skipped": skipped,
        "skipped_count": len(skipped),
        "course": course or "uncategorized",
    })


@app.delete("/files/<path:filename>")
def delete_file(filename):
    _bump_activity()
    safe_name = secure_filename(filename)
    ext = Path(safe_name).suffix.lower().lstrip(".")
    if ext not in SUPPORTED_EXTS:
        return jsonify({"error": "unsupported file type"}), 400
    course = (request.args.get("course") or "").strip()
    target_dir = _course_upload_dir(course) if course else COURSES_DIR / "uncategorized"
    target = target_dir / safe_name
    if safe_name and target.is_file() and target.suffix.lower().lstrip(".") in SUPPORTED_EXTS:
        source = _rel_source(target)
        target.unlink()
        copyrighted = load_copyrighted()
        if source in copyrighted:
            copyrighted.discard(source)
            save_copyrighted(copyrighted)
        return jsonify({"deleted": safe_name})
    return jsonify({"error": "not found"}), 404


@app.post("/files/copyright")
def set_file_copyright():
    """Mark an uploaded file as copyrighted (or not): the chat UI then never shows its text."""
    _bump_activity()
    data = request.get_json(force=True) or {}
    course = (data.get("course") or "").strip()
    safe_name = secure_filename(data.get("name") or "")
    target_dir = _course_upload_dir(course) if course else COURSES_DIR / "uncategorized"
    target = target_dir / safe_name
    if not safe_name or not target.is_file():
        return jsonify({"error": "not found"}), 404
    source = _rel_source(target)
    flag = bool(data.get("copyrighted"))
    copyrighted = load_copyrighted()
    if flag:
        copyrighted.add(source)
    else:
        copyrighted.discard(source)
    save_copyrighted(copyrighted)
    # Already indexed chunks get the tag now; new ones get it at their ingest
    try:
        set_copyrighted_in_qdrant(source, flag)
    except Exception as e:
        return jsonify({"error": f"saved, but updating the index failed: {e}", "source": source}), 500
    runs.log_event("copyright", source=source, copyrighted=flag)
    return jsonify({"source": source, "copyrighted": flag})


_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Ingest</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 760px; margin: 2rem auto; padding: 0 1rem; }
  button { font-size: 1rem; padding: 0.5rem 1rem; cursor: pointer; }
  button:disabled { cursor: not-allowed; opacity: 0.6; }
  h2 { margin-top: 2.5rem; }
  #status { margin: 1rem 0; font-weight: 600; }
  .idle { color: #666; }
  .running { color: #b8860b; }
  .done { color: #2a7a2a; }
  .error { color: #c0392b; }
  .interrupted { color: #b35c00; }
  pre { background: #111; color: #ddd; padding: 1rem; height: 16rem; overflow-y: auto;
        white-space: pre-wrap; word-break: break-word; border-radius: 6px; }
  /* Long lists scroll inside a panel instead of stretching the page */
  .panel { border: 1px solid #ddd; border-radius: 8px; overflow: hidden; margin-top: 0.75rem; }
  .panel-bar { display: flex; gap: 0.75rem; align-items: center; justify-content: space-between;
               padding: 0.45rem 0.6rem; background: #f7f7f8; border-bottom: 1px solid #ddd;
               font-size: 0.85rem; color: #555; }
  .panel-bar input[type=search] { font-size: 0.9rem; padding: 0.3rem 0.5rem; border: 1px solid #ccc;
                                  border-radius: 5px; width: 15rem; max-width: 55%; }
  .scroll { max-height: 22rem; overflow-y: auto; }
  .empty { padding: 0.8rem 0.6rem; color: #888; font-size: 0.9rem; }
  .dir { color: #999; }
  ul#file-list { list-style: none; padding: 0; margin: 0; }
  ul#file-list li { display: flex; align-items: center; justify-content: space-between; gap: 0.8rem;
                    padding: 0.35rem 0.6rem; border-bottom: 1px solid #f0f0f0; font-size: 0.9rem; }
  ul#file-list li:last-child { border-bottom: none; }
  ul#file-list li:hover { background: #fafafa; }
  ul#file-list li .name { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  ul#file-list li button { font-size: 0.8rem; padding: 0.15rem 0.55rem; }
  ul#file-list li .file-actions { display: flex; align-items: center; gap: 0.8rem; flex-shrink: 0; }
  ul#file-list li label { font-size: 0.85rem; white-space: nowrap; }
  table.data { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
  table.data th { position: sticky; top: 0; background: #f7f7f8; text-align: left; padding: 0.45rem 0.6rem;
                  border-bottom: 1px solid #ddd; font-weight: 600; white-space: nowrap;
                  cursor: pointer; user-select: none; }
  table.data th:hover { color: #000; background: #efeff1; }
  table.data td { padding: 0.35rem 0.6rem; border-bottom: 1px solid #f0f0f0; }
  table.data tr:last-child td { border-bottom: none; }
  table.data tbody tr:hover td { background: #fafafa; }
  table.data .num { text-align: right; font-variant-numeric: tabular-nums; width: 5rem; }
  table.data .course { width: 7rem; white-space: nowrap; }
  table.data td.src { word-break: break-word; }
  .copyright-label { display: inline-block; margin-top: 0.5rem; font-size: 0.9rem; }
  .hint { color: #666; }
  #upload-msg { margin-top: 0.5rem; font-size: 0.9rem; }
  .course-row { display: flex; gap: 0.5rem; align-items: center; margin-bottom: 0.5rem; }
  .course-row input { font-size: 1rem; padding: 0.35rem 0.5rem; flex: 1; }
  .course-row button { font-size: 0.9rem; }
  #course-list { list-style: none; padding: 0; }
  #course-list li { display: flex; align-items: center; justify-content: space-between;
                    padding: 0.4rem 0.6rem; border-bottom: 1px solid #eee; }
  #course-select { font-size: 1rem; padding: 0.35rem 0.5rem; }
  .tag { font-size: 0.75rem; color: #555; background: #eef1f6; border-radius: 999px; padding: 0.1rem 0.5rem; margin-left: 0.5rem; }
  /* Notes repositories */
  table.data.plain th { cursor: default; }
  table.data.plain th:hover { background: #f7f7f8; color: inherit; }
  table.repos td { vertical-align: top; }
  table.repos .actions { white-space: nowrap; text-align: right; width: 1%; }
  table.repos .actions button { font-size: 0.8rem; padding: 0.15rem 0.55rem; }
  .commit { font-size: 0.85rem; }
  .commit .sha { font-family: ui-monospace, Consolas, monospace; color: #555; }
  .meta { font-size: 0.78rem; color: #888; margin-top: 0.15rem; }
  .err { font-size: 0.8rem; color: #c0392b; margin-top: 0.15rem; word-break: break-word; }
  .notice { background: #fff8e1; border: 1px solid #f0d58a; border-radius: 6px; padding: 0.5rem 0.7rem;
            font-size: 0.88rem; margin-top: 0.6rem; }
  .repo-form { display: grid; grid-template-columns: 9rem 1fr; gap: 0.45rem 0.7rem; align-items: center;
               margin-top: 0.75rem; padding: 0.8rem; border: 1px solid #ddd; border-radius: 8px; }
  .repo-form label { font-size: 0.9rem; color: #444; }
  .repo-form input { font-size: 0.95rem; padding: 0.35rem 0.5rem; width: 100%; box-sizing: border-box; }
  .repo-form .help { grid-column: 2; font-size: 0.78rem; color: #777; margin-top: -0.25rem; }
  .repo-form .buttons { grid-column: 2; display: flex; gap: 0.5rem; align-items: center; }
  .repo-form .buttons button { font-size: 0.9rem; }
  #repo-msg { font-size: 0.88rem; }
  @media (max-width: 560px) { .repo-form { grid-template-columns: 1fr; } .repo-form .help, .repo-form .buttons { grid-column: 1; } }
</style>
</head>
<body>
  <h1>Ingest</h1>

  <h2>Notes repositories</h2>
  <p>Public GitHub repositories with course notes in the Quarto course layout (<code>index.qmd</code>,
  <code>pages/*.qmd</code>, <code>_includes/**/*.md</code>, and any PDFs). Each one is a course: you can
  upload more sources into it below. They are checked for new commits every 10 minutes, and changes are
  ingested automatically.</p>
  <div id="repo-pending" class="notice" hidden>Repository changes are not ingested yet. The next check
    (within 10 minutes) ingests them, or click <b>Start ingest</b>.</div>
  <div class="panel">
    <div class="scroll">
      <table id="repos-table" class="data plain repos">
        <thead><tr><th>Course</th><th>Repository</th><th>Latest commit</th><th></th></tr></thead>
        <tbody></tbody>
      </table>
      <div id="repos-empty" class="empty" hidden>No repositories yet.</div>
    </div>
  </div>
  <div class="repo-form" id="repo-form">
    <label for="repo-url">GitHub link</label>
    <input type="url" id="repo-url" placeholder="https://github.com/owner/repo">
    <label for="repo-course">Course</label>
    <input type="text" id="repo-course" placeholder="e.g. TMA4411">
    <label for="repo-site">Published site</label>
    <input type="url" id="repo-site" placeholder="filled in from the link (GitHub Pages)">
    <div class="help">Where the notes are published, so answers can link to the right page. Leave empty for GitHub Pages.</div>
    <label for="repo-exclude">Files to skip</label>
    <input type="text" id="repo-exclude" placeholder="e.g. *fasit*.pdf, assets/old/*">
    <div class="help">Comma-separated patterns, matched against the path in the repo or the file name.</div>
    <div class="buttons">
      <button id="repo-submit" onclick="submitRepo()">Add repository</button>
      <button id="repo-cancel" onclick="resetRepoForm()" hidden>Cancel</button>
      <span id="repo-msg"></span>
    </div>
  </div>

  <h2>Courses</h2>
  <p>Create a course folder. Uploaded files go into the selected course.</p>
  <div class="course-row">
    <input type="text" id="new-course" placeholder="e.g. ma1101">
    <button onclick="createCourse()">Create</button>
  </div>
  <ul id="course-list"></ul>

  <h2>Upload File</h2>
  <p>Select a course, then upload a PDF, Markdown, or QMD file. It will be ingested into that course.</p>
  <select id="course-select"><option value="">-- select course --</option></select><br><br>
  <input type="file" id="file-input" multiple accept=".pdf,.md,.qmd,application/pdf,text/markdown">
  <button onclick="uploadFile()">Upload</button><br>
  <label class="copyright-label"><input type="checkbox" id="upload-copyrighted"> Copyrighted material
    <span class="hint">(answers may use it, but students can't open its text)</span></label>
  <div id="upload-msg"></div>
  <div class="panel">
    <div class="panel-bar">
      <span id="files-count"></span>
      <input type="search" id="files-filter" placeholder="Filter files" oninput="renderFiles()">
    </div>
    <div class="scroll"><ul id="file-list"></ul></div>
  </div>

  <h2>Run</h2>
  <p>Incremental by default: only new/changed files are re-chunked and re-embedded, unchanged
  files (and unchanged uploaded files) are skipped, removed files are cleaned up from Qdrant.</p>
  <label><input type="checkbox" id="force-full"> Force full rebuild (deletes and re-creates the whole collection, re-processes everything)</label><br><br>
  <button id="run-btn" onclick="runIngest()">Start ingest</button>
  <div id="status" class="idle">idle</div>
  <pre id="log"></pre>

  <h2>Indexed Sources</h2>
  <p>Everything currently stored in Qdrant, with chunk counts.</p>
  <div id="sources-summary" class="hint" style="font-size: 0.85rem;"></div>
  <div class="panel">
    <div class="panel-bar">
      <span id="sources-count"></span>
      <input type="search" id="sources-filter" placeholder="Filter sources" oninput="renderSources()">
    </div>
    <div class="scroll">
      <table id="sources-table" class="data">
        <thead>
          <tr>
            <th data-key="source" onclick="sortSources('source')">Source</th>
            <th data-key="chunks" class="num" onclick="sortSources('chunks')">Chunks</th>
            <th data-key="course" class="course" onclick="sortSources('course')">Course</th>
          </tr>
        </thead>
        <tbody></tbody>
      </table>
      <div id="sources-empty" class="empty" hidden></div>
    </div>
  </div>

<script>
const btn = document.getElementById('run-btn');
const statusEl = document.getElementById('status');
const logEl = document.getElementById('log');
const fileInputEl = document.getElementById('file-input');
const fileListEl = document.getElementById('file-list');
const uploadMsgEl = document.getElementById('upload-msg');
const forceFullEl = document.getElementById('force-full');
const courseSelect = document.getElementById('course-select');
const courseListEl = document.getElementById('course-list');
const newCourseEl = document.getElementById('new-course');

// API paths are relative: the page lives at .../ingest/, so they go through the same route (and login)
async function loadCourses() {
  const res = await fetch('admin/courses');
  const names = await res.json();
  const selected = courseSelect.value;
  courseListEl.innerHTML = '';
  courseSelect.innerHTML = '<option value="">-- select course --</option>';
  for (const name of names) {
    const li = document.createElement('li');
    const label = document.createElement('span');
    label.textContent = name;
    const repo = reposData.find(r => r.course === name);
    if (repo) {
      const tag = document.createElement('span');
      tag.className = 'tag';
      tag.textContent = 'from repo ' + repoName(repo.url);
      label.appendChild(tag);
    }
    const delBtn = document.createElement('button');
    delBtn.textContent = 'Delete';
    delBtn.onclick = () => deleteCourse(name);
    li.appendChild(label);
    li.appendChild(delBtn);
    courseListEl.appendChild(li);

    const opt = document.createElement('option');
    opt.value = name;
    opt.textContent = name;
    courseSelect.appendChild(opt);
  }
  if (names.includes(selected)) courseSelect.value = selected;
}

/* --- Notes repositories --- */
let reposData = [];
let editingCourse = null;

function repoName(url) {
  return url.replace('https://github.com/', '');
}

function ago(ts) {
  if (!ts) return '';
  const s = Math.round((Date.now() - new Date(ts)) / 1000);
  if (s < 90) return 'just now';
  if (s < 5400) return Math.round(s / 60) + ' min ago';
  if (s < 129600) return Math.round(s / 3600) + ' h ago';
  return Math.round(s / 86400) + ' days ago';
}

async function loadRepos() {
  try {
    const res = await fetch('admin/repos');
    const data = await res.json();
    reposData = data.repos || [];
    document.getElementById('repo-pending').hidden = !data.pending_ingest;
  } catch (e) {
    reposData = [];
  }
  renderRepos();
}

function renderRepos() {
  const tbody = document.querySelector('#repos-table tbody');
  tbody.innerHTML = '';
  document.getElementById('repos-empty').hidden = reposData.length > 0;
  for (const r of reposData) {
    const tr = document.createElement('tr');
    const tdCourse = document.createElement('td');
    tdCourse.textContent = r.course;

    const tdRepo = document.createElement('td');
    const a = document.createElement('a');
    a.href = r.url;
    a.target = '_blank';
    a.rel = 'noopener';
    a.textContent = repoName(r.url);
    tdRepo.appendChild(a);
    const meta = document.createElement('div');
    meta.className = 'meta';
    meta.textContent = (r.site_url ? 'site: ' + r.site_url.replace(/^https?:[/][/]/, '') : 'no published site')
      + (r.exclude && r.exclude.length ? ' · skips: ' + r.exclude.join(', ') : '');
    tdRepo.appendChild(meta);

    const tdCommit = document.createElement('td');
    tdCommit.className = 'commit';
    const lc = r.last_check || {};
    if (lc.commit) {
      const sha = document.createElement('span');
      sha.className = 'sha';
      sha.textContent = lc.commit + ' ';
      tdCommit.append(sha, document.createTextNode(lc.subject || ''));
    }
    const when = document.createElement('div');
    when.className = 'meta';
    when.textContent = (lc.date ? 'committed ' + ago(lc.date) : '') + (lc.at ? (lc.date ? ' · ' : '') + 'checked ' + ago(lc.at) : '');
    tdCommit.appendChild(when);
    if (lc.error) {
      const err = document.createElement('div');
      err.className = 'err';
      err.textContent = lc.error;
      tdCommit.appendChild(err);
    }

    const tdActions = document.createElement('td');
    tdActions.className = 'actions';
    const edit = document.createElement('button');
    edit.textContent = 'Edit';
    edit.onclick = () => editRepo(r.course);
    const del = document.createElement('button');
    del.textContent = 'Remove';
    del.onclick = () => removeRepo(r.course);
    tdActions.append(edit, ' ', del);

    tr.append(tdCourse, tdRepo, tdCommit, tdActions);
    tbody.appendChild(tr);
  }
}

function resetRepoForm() {
  editingCourse = null;
  for (const id of ['repo-url', 'repo-course', 'repo-site', 'repo-exclude']) document.getElementById(id).value = '';
  document.getElementById('repo-course').disabled = false;
  document.getElementById('repo-submit').textContent = 'Add repository';
  document.getElementById('repo-cancel').hidden = true;
  document.getElementById('repo-msg').textContent = '';
}

function editRepo(course) {
  const r = reposData.find(x => x.course === course);
  if (!r) return;
  editingCourse = course;
  document.getElementById('repo-url').value = r.url;
  document.getElementById('repo-course').value = r.course;
  document.getElementById('repo-course').disabled = true;  // a different course is a different repo entry
  document.getElementById('repo-site').value = r.site_url || '';
  document.getElementById('repo-exclude').value = (r.exclude || []).join(', ');
  document.getElementById('repo-submit').textContent = 'Save changes';
  document.getElementById('repo-cancel').hidden = false;
  document.getElementById('repo-msg').textContent = 'Editing ' + course + '. A new link keeps the same clone; files that are the same there are not re-embedded.';
  document.getElementById('repo-form').scrollIntoView({ behavior: 'smooth', block: 'center' });
}

async function submitRepo() {
  const msg = document.getElementById('repo-msg');
  const submit = document.getElementById('repo-submit');
  const body = {
    url: document.getElementById('repo-url').value.trim(),
    site_url: document.getElementById('repo-site').value.trim(),
    exclude: document.getElementById('repo-exclude').value,
  };
  if (!editingCourse) body.course = document.getElementById('repo-course').value.trim();
  if (!body.url || (!editingCourse && !body.course)) {
    msg.textContent = 'A GitHub link and a course name are needed.';
    return;
  }
  submit.disabled = true;
  msg.textContent = editingCourse ? 'Saving...' : 'Cloning the repository...';
  const res = await fetch(editingCourse ? 'admin/repos/' + encodeURIComponent(editingCourse) : 'admin/repos', {
    method: editingCourse ? 'PATCH' : 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  submit.disabled = false;
  if (!res.ok) {
    msg.textContent = 'Error: ' + (data.error || res.statusText);
    return;
  }
  const done = editingCourse ? 'Saved.' : 'Added ' + body.course + '.';
  resetRepoForm();
  msg.textContent = done;
  reposData = data.repos || [];
  document.getElementById('repo-pending').hidden = !data.pending_ingest;
  renderRepos();
  loadCourses();
}

async function removeRepo(course) {
  if (!confirm('Remove the repository for ' + course + '? Its notes leave the index at the next ingest. '
      + 'Files you uploaded to ' + course + ' stay.')) return;
  const res = await fetch('admin/repos/' + encodeURIComponent(course), { method: 'DELETE' });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    alert(data.error || 'Failed');
    return;
  }
  if (editingCourse === course) resetRepoForm();
  reposData = data.repos || [];
  document.getElementById('repo-pending').hidden = !data.pending_ingest;
  renderRepos();
  loadCourses();
}

async function createCourse() {
  const name = newCourseEl.value.trim();
  if (!name) return;
  const res = await fetch('admin/courses', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name }),
  });
  const body = await res.json();
  if (!res.ok) {
    alert(body.error || 'Failed');
    return;
  }
  newCourseEl.value = '';
  loadCourses();
}

async function deleteCourse(name) {
  if (!confirm('Delete course ' + name + '? Only works if empty.')) return;
  const res = await fetch('admin/courses/' + encodeURIComponent(name), { method: 'DELETE' });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    alert(body.error || 'Failed');
    return;
  }
  loadCourses();
}

async function loadFiles() {
  const course = courseSelect.value;
  const qs = course ? '?course=' + encodeURIComponent(course) : '';
  const res = await fetch('files' + qs);
  filesData = await res.json();
  renderFiles();
}

let filesData = [];

// "IMAx3011/book.md" -> muted "IMAx3011/" + "book.md"
function pathLabel(el, path) {
  const cut = path.lastIndexOf('/') + 1;
  el.textContent = '';
  if (cut) {
    const dir = document.createElement('span');
    dir.className = 'dir';
    dir.textContent = path.slice(0, cut);
    el.appendChild(dir);
  }
  el.appendChild(document.createTextNode(path.slice(cut)));
  el.title = path;
}

function renderFiles() {
  const q = document.getElementById('files-filter').value.trim().toLowerCase();
  const files = filesData.filter(f => !q || f.name.toLowerCase().includes(q));
  const course = courseSelect.value;
  document.getElementById('files-count').textContent =
    (q ? files.length + ' of ' : '') + filesData.length + ' file' + (filesData.length === 1 ? '' : 's')
    + (course ? ' in ' + course : '');
  fileListEl.innerHTML = '';
  if (!files.length) {
    const li = document.createElement('li');
    li.className = 'empty';
    li.textContent = q ? 'No files match.' : 'No uploaded files' + (course ? ' in this course.' : '.');
    fileListEl.appendChild(li);
    return;
  }
  for (const file of files) {
    const li = document.createElement('li');
    const label = document.createElement('span');
    label.className = 'name';
    pathLabel(label, file.name);
    const actions = document.createElement('span');
    actions.className = 'file-actions';
    const crLabel = document.createElement('label');
    const cr = document.createElement('input');
    cr.type = 'checkbox';
    cr.checked = !!file.copyrighted;
    cr.onchange = () => setCopyright(file.name, cr);
    crLabel.appendChild(cr);
    crLabel.appendChild(document.createTextNode(' Copyrighted'));
    const delBtn = document.createElement('button');
    delBtn.textContent = 'Delete';
    delBtn.onclick = () => deleteFile(file.name);
    actions.appendChild(crLabel);
    actions.appendChild(delBtn);
    li.appendChild(label);
    li.appendChild(actions);
    fileListEl.appendChild(li);
  }
}

async function setCopyright(name, checkbox) {
  checkbox.disabled = true;
  const res = await fetch('files/copyright', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ course: courseSelect.value, name, copyrighted: checkbox.checked }),
  });
  const body = await res.json().catch(() => ({}));
  checkbox.disabled = false;
  if (!res.ok) {
    alert(body.error || 'Failed');
    if (!body.source) checkbox.checked = !checkbox.checked; // not saved at all
  }
}

async function uploadFile() {
  const files = fileInputEl.files;
  const course = courseSelect.value;
  if (!course) {
    uploadMsgEl.textContent = 'Please select a course first.';
    return;
  }
  if (!files || !files.length) {
    uploadMsgEl.textContent = 'Please select at least one file first.';
    return;
  }
  const formData = new FormData();
  for (let i = 0; i < files.length; i++) {
    formData.append('file', files[i]);
  }
  formData.append('course', course);
  formData.append('copyrighted', document.getElementById('upload-copyrighted').checked ? 'true' : 'false');
  const res = await fetch('files', { method: 'POST', body: formData });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    uploadMsgEl.textContent = 'Error: ' + (body.error || res.statusText);
    return;
  }
  const msgParts = [];
  if (body.uploaded_count) {
    msgParts.push(body.uploaded_count + ' file(s) uploaded to ' + body.course);
  }
  if (body.skipped_count) {
    msgParts.push(body.skipped_count + ' skipped (' + body.skipped.join(', ') + ')');
  }
  uploadMsgEl.textContent = msgParts.join(' | ');
  fileInputEl.value = '';
  document.getElementById('upload-copyrighted').checked = false;
  loadFiles();
}

async function deleteFile(name) {
  const course = courseSelect.value;
  const qs = course ? '?course=' + encodeURIComponent(course) : '';
  await fetch('files/' + encodeURIComponent(name) + qs, { method: 'DELETE' });
  loadFiles();
}

courseSelect.addEventListener('change', () => {
  loadFiles();
  loadSources();
});

async function init() {
  await loadRepos();  // first: the course list marks the courses that come from a repo
  await loadCourses();
  loadFiles();
  loadSources();
}
init();

async function loadSources() {
  const course = courseSelect.value;
  const qs = course ? '?course=' + encodeURIComponent(course) : '';
  const summary = document.getElementById('sources-summary');
  try {
    const res = await fetch('sources' + qs);
    const data = await res.json();
    if (!data.indexed) {
      summary.textContent = '';
      sourcesData = [];
      sourcesTotalChunks = 0;
      renderSources('Nothing indexed yet. Run an ingest first.');
      return;
    }
    summary.textContent = `Collection ${data.collection || '-'} · dim ${data.embedding_dim || '-'}`;
    sourcesData = data.sources;
    sourcesTotalChunks = data.total_chunks;
    renderSources();
  } catch (e) {
    summary.textContent = 'Failed to load sources: ' + e.message;
  }
}

let sourcesData = [];
let sourcesTotalChunks = 0;
let sourcesSort = { key: 'source', dir: 1 };

function sortSources(key) {
  // A new column sorts ascending, except chunks: biggest first is the useful view
  sourcesSort = sourcesSort.key === key ? { key, dir: -sourcesSort.dir } : { key, dir: key === 'chunks' ? -1 : 1 };
  renderSources();
}

function renderSources(emptyMsg) {
  const q = document.getElementById('sources-filter').value.trim().toLowerCase();
  const rows = sourcesData.filter(s => !q || (s.source + ' ' + (s.course || '')).toLowerCase().includes(q));
  const { key, dir } = sourcesSort;
  rows.sort((a, b) => {
    const x = key === 'chunks' ? a.chunks : String(a[key] || '').toLowerCase();
    const y = key === 'chunks' ? b.chunks : String(b[key] || '').toLowerCase();
    return (x < y ? -1 : x > y ? 1 : 0) * dir;
  });

  const course = courseSelect.value;
  const chunks = q ? rows.reduce((n, s) => n + s.chunks, 0) : sourcesTotalChunks;
  document.getElementById('sources-count').textContent =
    (q ? rows.length + ' of ' : '') + sourcesData.length + ' sources · ' + chunks + ' chunks'
    + (course ? ' in ' + course : '');

  for (const th of document.querySelectorAll('#sources-table th')) {
    const base = th.textContent.replace(/ [▲▼]$/, '');
    th.textContent = base + (th.dataset.key === key ? (dir === 1 ? ' ▲' : ' ▼') : '');
  }

  const tbody = document.querySelector('#sources-table tbody');
  const empty = document.getElementById('sources-empty');
  tbody.innerHTML = '';
  empty.hidden = rows.length > 0;
  empty.textContent = emptyMsg || (q ? 'No sources match.' : 'No sources in this course.');
  for (const item of rows) {
    const tr = document.createElement('tr');
    const tdSrc = document.createElement('td');
    tdSrc.className = 'src';
    pathLabel(tdSrc, item.source);
    const tdChunks = document.createElement('td');
    tdChunks.className = 'num';
    tdChunks.textContent = item.chunks;
    const tdCourse = document.createElement('td');
    tdCourse.className = 'course';
    tdCourse.textContent = item.course || '-';
    tr.append(tdSrc, tdChunks, tdCourse);
    tbody.appendChild(tr);
  }
}

async function runIngest() {
  const res = await fetch('run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ force_full: forceFullEl.checked }),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    alert(body.error || 'Failed to start ingest');
    return;
  }
  // The run starts after the repositories are pulled: keep polling until it shows up
  awaitingRun = { before: lastStartedAt, until: Date.now() + 120000 };
  statusEl.textContent = 'pulling the notes repositories...';
  statusEl.className = 'running';
  btn.disabled = true;
  poll();
}

let lastStartedAt = null;
let lastStatus = null;
let awaitingRun = null;

function render(state) {
  statusEl.textContent = state.status
    + (state.started_at ? ' (started ' + state.started_at + ')' : '')
    + (state.finished_at ? ' (finished ' + state.finished_at + ')' : '');
  statusEl.className = state.status;
  logEl.textContent = state.log + (state.error ? '\\n\\n' + state.error : '');
  logEl.scrollTop = logEl.scrollHeight;
  btn.disabled = state.status === 'running';
}

async function poll() {
  const res = await fetch('status');
  const state = await res.json();
  const waiting = awaitingRun && state.started_at === awaitingRun.before && Date.now() < awaitingRun.until;
  if (!waiting) {
    awaitingRun = null;
    render(state);
  }
  // A run just ended: the index and the repositories' state changed
  if (lastStatus === 'running' && state.status !== 'running') {
    loadRepos();
    loadSources();
  }
  lastStatus = state.status;
  lastStartedAt = state.started_at;
  if (state.status === 'running' || waiting) {
    setTimeout(poll, 1500);
  }
}

poll();
</script>
</body>
</html>
"""
