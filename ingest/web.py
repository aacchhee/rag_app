"""
Minimal web UI for the ingest service.

Wraps ingest.ingest.main() so a re-index can be triggered from a browser
instead of a shell, with live status/log output. Runs as a small Flask app;
the actual ingest logic in ingest/ingest.py is untouched.
"""

import contextlib
import io
import json
import os
import signal
import sys
import threading
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

from flask import Flask, Response, jsonify, request
from werkzeug.utils import secure_filename

from ingest.ingest import _load_manifest, COURSES_DIR, list_courses, main as run_ingest

# --- Event history (JSONL) ---
HISTORY_FILE = COURSES_DIR.parent / "ingest_history.jsonl"
_MAX_HISTORY_RETENTION_DAYS = 90


def _log_event(event_type: str, **extra) -> None:
    entry = {
        "t": datetime.now(timezone.utc).isoformat(),
        "type": event_type,
        **extra,
    }
    try:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(HISTORY_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB per upload

_LOCK = threading.Lock()
_state = {
    "status": "idle",  # idle | running | done | error
    "started_at": None,
    "finished_at": None,
    "error": None,
    "log": "",
}

# Auto-exit after 30 minutes of inactivity once a job completes.
_last_request_time = datetime.now(timezone.utc)
_AUTO_EXIT_DELAY_SEC = int(os.getenv("INGEST_AUTO_EXIT_DELAY_MIN", "30")) * 60


class _LogStream(io.StringIO):
    def write(self, s):
        with _LOCK:
            _state["log"] += s
        return super().write(s)


def _bump_activity():
    global _last_request_time
    _last_request_time = datetime.now(timezone.utc)


def _schedule_auto_exit():
    def _watchdog():
        while True:
            threading.Event().wait(timeout=_AUTO_EXIT_DELAY_SEC)
            with _LOCK:
                status = _state["status"]
                idle_for = (datetime.now(timezone.utc) - _last_request_time).total_seconds()
            if status != "running" and idle_for >= _AUTO_EXIT_DELAY_SEC:
                os.kill(os.getpid(), signal.SIGTERM)
    threading.Thread(target=_watchdog, daemon=True).start()


_schedule_auto_exit()


def _run_job(force_full: bool = False):
    _log_event("run_start", force_full=force_full)
    with _LOCK:
        _state.update(
            status="running",
            started_at=datetime.now(timezone.utc).isoformat(),
            finished_at=None,
            error=None,
            log="",
        )

    try:
        with contextlib.redirect_stdout(_LogStream()):
            run_ingest(force_full=force_full)
        with _LOCK:
            _state["status"] = "done"
        _log_event("run_done")
    except Exception:
        with _LOCK:
            _state["status"] = "error"
            _state["error"] = traceback.format_exc()
        _log_event("run_error")
    finally:
        with _LOCK:
            _state["finished_at"] = datetime.now(timezone.utc).isoformat()


@app.get("/")
def index():
    _bump_activity()
    return Response(_PAGE, mimetype="text/html")


@app.get("/history")
def get_history():
    _bump_activity()
    since = request.args.get("since")
    until = request.args.get("until")
    cutoff = datetime.now(timezone.utc) - timedelta(days=_MAX_HISTORY_RETENTION_DAYS)

    events = []
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                    t = datetime.fromisoformat(ev.get("t", "1970-01-01T00:00:00+00:00"))
                    if t < cutoff:
                        continue
                    if since and t < datetime.fromisoformat(since):
                        continue
                    if until and t > datetime.fromisoformat(until):
                        continue
                    events.append(ev)
                except (ValueError, TypeError):
                    continue
    return jsonify(events=list(reversed(events[-10000:])))


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
            "source": src,
            "chunks": len(meta.get("chunk_ids", [])),
            "hash": meta.get("hash", "")[:16] + "…",
            "course": meta.get("course"),
        }
        for src, meta in sorted(sources.items())
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
    with _LOCK:
        return jsonify(dict(_state))


@app.post("/run")
def trigger():
    with _LOCK:
        if _state["status"] == "running":
            return jsonify({"error": "ingest already running"}), 409
    force_full = bool((request.get_json(silent=True) or {}).get("force_full"))
    threading.Thread(target=_run_job, args=(force_full,), daemon=True).start()
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
    safe = secure_filename(name)
    target = COURSES_DIR / safe
    if not safe or not target.is_dir():
        return jsonify({"error": "not found"}), 404
    # Only delete if empty
    files = list(target.rglob("*"))
    if any(p.is_file() for p in files):
        return jsonify({"error": "course is not empty"}), 400
    try:
        target.rmdir()
        return jsonify({"deleted": safe})
    except OSError:
        return jsonify({"error": "could not delete"}), 500


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
    names = sorted(
        p.name
        for p in d.rglob("*")
        if p.is_file() and p.suffix.lower().lstrip(".") in SUPPORTED_EXTS
    )
    return jsonify(names)


@app.post("/files")
def upload_file():
    _bump_activity()
    file = request.files.get("file")
    if file is None or not file.filename:
        return jsonify({"error": "no file provided"}), 400

    filename = secure_filename(file.filename)
    ext = Path(filename).suffix.lower().lstrip(".")
    if ext not in SUPPORTED_EXTS:
        return jsonify({"error": f"only {', '.join(sorted(SUPPORTED_EXTS))} files are allowed"}), 400

    if ext == "pdf":
        head = file.stream.read(5)
        file.stream.seek(0)
        if head != b"%PDF-":
            return jsonify({"error": "file does not look like a PDF"}), 400

    course = (request.form.get("course") or "").strip()
    target_dir = _course_upload_dir(course) if course else COURSES_DIR / "uncategorized"
    target_dir.mkdir(parents=True, exist_ok=True)
    file.save(target_dir / filename)
    return jsonify({"saved": filename, "course": course or "uncategorized"})


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
        target.unlink()
        return jsonify({"deleted": safe_name})
    return jsonify({"error": "not found"}), 404


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
  pre { background: #111; color: #ddd; padding: 1rem; height: 420px; overflow-y: auto;
        white-space: pre-wrap; word-break: break-word; border-radius: 6px; }
  ul#file-list { list-style: none; padding: 0; }
  ul#file-list li { display: flex; align-items: center; justify-content: space-between;
                    padding: 0.4rem 0.6rem; border-bottom: 1px solid #ddd; }
  ul#file-list li button { font-size: 0.85rem; padding: 0.2rem 0.6rem; }
  #upload-msg { margin-top: 0.5rem; font-size: 0.9rem; }
  .course-row { display: flex; gap: 0.5rem; align-items: center; margin-bottom: 0.5rem; }
  .course-row input { font-size: 1rem; padding: 0.35rem 0.5rem; flex: 1; }
  .course-row button { font-size: 0.9rem; }
  #course-list { list-style: none; padding: 0; }
  #course-list li { display: flex; align-items: center; justify-content: space-between;
                    padding: 0.4rem 0.6rem; border-bottom: 1px solid #eee; }
  #course-select { font-size: 1rem; padding: 0.35rem 0.5rem; }
</style>
</head>
<body>
  <h1>Ingest</h1>

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
  <input type="file" id="file-input" accept=".pdf,.md,.qmd,application/pdf,text/markdown">
  <button onclick="uploadFile()">Upload</button>
  <div id="upload-msg"></div>
  <ul id="file-list"></ul>

  <h2>Run</h2>
  <p>Incremental by default: only new/changed files are re-chunked and re-embedded, unchanged
  files (and unchanged uploaded files) are skipped, removed files are cleaned up from Qdrant.</p>
  <label><input type="checkbox" id="force-full"> Force full rebuild (deletes and re-creates the whole collection, re-processes everything)</label><br><br>
  <button id="run-btn" onclick="runIngest()">Start ingest</button>
  <div id="status" class="idle">idle</div>
  <pre id="log"></pre>

  <h2>Indexed Sources</h2>
  <p>Everything currently stored in Qdrant, with chunk counts.</p>
  <div id="sources-summary" style="font-size: 0.9rem; color: #555; margin-bottom: 0.5rem;"></div>
  <table id="sources-table" style="width: 100%; border-collapse: collapse; font-size: 0.9rem;">
    <thead>
      <tr style="border-bottom: 2px solid #ccc; text-align: left;">
        <th style="padding: 0.4rem;">Source</th>
        <th style="padding: 0.4rem; width: 80px;">Chunks</th>
        <th style="padding: 0.4rem; width: 100px;">Course</th>
      </tr>
    </thead>
    <tbody></tbody>
  </table>

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

async function loadCourses() {
  const res = await fetch('/admin/courses');
  const names = await res.json();
  courseListEl.innerHTML = '';
  courseSelect.innerHTML = '<option value="">-- select course --</option>';
  for (const name of names) {
    const li = document.createElement('li');
    const label = document.createElement('span');
    label.textContent = name;
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
}

async function createCourse() {
  const name = newCourseEl.value.trim();
  if (!name) return;
  const res = await fetch('/admin/courses', {
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
  const res = await fetch('/admin/courses/' + encodeURIComponent(name), { method: 'DELETE' });
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
  const res = await fetch('/files' + qs);
  const names = await res.json();
  fileListEl.innerHTML = '';
  for (const name of names) {
    const li = document.createElement('li');
    const label = document.createElement('span');
    label.textContent = name;
    const delBtn = document.createElement('button');
    delBtn.textContent = 'Delete';
    delBtn.onclick = () => deleteFile(name);
    li.appendChild(label);
    li.appendChild(delBtn);
    fileListEl.appendChild(li);
  }
}

async function uploadFile() {
  const file = fileInputEl.files[0];
  const course = courseSelect.value;
  if (!course) {
    uploadMsgEl.textContent = 'Please select a course first.';
    return;
  }
  if (!file) {
    uploadMsgEl.textContent = 'Please select a file first.';
    return;
  }
  const formData = new FormData();
  formData.append('file', file);
  formData.append('course', course);
  const res = await fetch('/files', { method: 'POST', body: formData });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    uploadMsgEl.textContent = 'Error: ' + (body.error || res.statusText);
    return;
  }
  uploadMsgEl.textContent = body.saved + ' uploaded to ' + body.course + '.';
  fileInputEl.value = '';
  loadFiles();
}

async function deleteFile(name) {
  const course = courseSelect.value;
  const qs = course ? '?course=' + encodeURIComponent(course) : '';
  await fetch('/files/' + encodeURIComponent(name) + qs, { method: 'DELETE' });
  loadFiles();
}

courseSelect.addEventListener('change', () => {
  loadFiles();
  loadSources();
});

async function init() {
  await loadCourses();
  loadFiles();
  loadSources();
}
init();

async function loadSources() {
  const course = courseSelect.value;
  const qs = course ? '?course=' + encodeURIComponent(course) : '';
  const tbody = document.querySelector('#sources-table tbody');
  const summary = document.getElementById('sources-summary');
  try {
    const res = await fetch('/sources' + qs);
    const data = await res.json();
    if (!data.indexed) {
      summary.textContent = 'Nothing indexed yet. Run an ingest first.';
      tbody.innerHTML = '';
      return;
    }
    const filterHint = course ? ` (course: ${course})` : '';
    summary.textContent = `Collection: ${data.collection || '-'} | dim=${data.embedding_dim || '-'} | ${data.total_sources} sources | ${data.total_chunks} chunks${filterHint}`;
    tbody.innerHTML = '';
    for (const item of data.sources) {
      const tr = document.createElement('tr');
      tr.style.borderBottom = '1px solid #eee';
      const tdSrc = document.createElement('td');
      tdSrc.style.padding = '0.4rem';
      tdSrc.textContent = item.source;
      const tdChunks = document.createElement('td');
      tdChunks.style.padding = '0.4rem';
      tdChunks.textContent = item.chunks;
      const tdCourse = document.createElement('td');
      tdCourse.style.padding = '0.4rem';
      tdCourse.textContent = item.course || '-';
      tr.appendChild(tdSrc);
      tr.appendChild(tdChunks);
      tr.appendChild(tdCourse);
      tbody.appendChild(tr);
    }
  } catch (e) {
    summary.textContent = 'Failed to load sources: ' + e.message;
  }
}

async function runIngest() {
  const res = await fetch('/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ force_full: forceFullEl.checked }),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    alert(body.error || 'Failed to start ingest');
    return;
  }
  poll();
}

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
  const res = await fetch('/status');
  const state = await res.json();
  render(state);
  if (state.status === 'running') {
    setTimeout(poll, 1500);
  }
}

poll();
</script>
</body>
</html>
"""
