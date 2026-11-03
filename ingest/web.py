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

from ingest import runs
from ingest.ingest import (
    _load_manifest, _rel_source, COURSES_DIR, list_courses, load_copyrighted, main as run_ingest,
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
        run_ingest(force_full=force_full, trigger="page", lock=lock)
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
  const res = await fetch('/files/copyright', {
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
  const res = await fetch('/files', { method: 'POST', body: formData });
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
  const summary = document.getElementById('sources-summary');
  try {
    const res = await fetch('/sources' + qs);
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
