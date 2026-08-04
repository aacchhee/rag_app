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

from flask import Flask, Response, jsonify, request
from werkzeug.utils import secure_filename

from ingest.ingest import _load_manifest, PDF_UPLOAD_DIR, main as run_ingest

# --- Event history (JSONL) ---
HISTORY_FILE = PDF_UPLOAD_DIR.parent / "ingest_history.jsonl"
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
    items = [
        {
            "source": src,
            "chunks": len(meta.get("chunk_ids", [])),
            "hash": meta.get("hash", "")[:16] + "…",
        }
        for src, meta in sorted(sources.items())
    ]
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


@app.get("/pdfs")
def list_pdfs():
    _bump_activity()
    PDF_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    return jsonify(sorted(p.name for p in PDF_UPLOAD_DIR.glob("*.pdf")))


@app.post("/pdfs")
def upload_pdf():
    _bump_activity()
    file = request.files.get("file")
    if file is None or not file.filename:
        return jsonify({"error": "no file provided"}), 400

    filename = secure_filename(file.filename)
    if not filename.lower().endswith(".pdf"):
        return jsonify({"error": "only .pdf files are allowed"}), 400

    head = file.stream.read(5)
    file.stream.seek(0)
    if head != b"%PDF-":
        return jsonify({"error": "file does not look like a PDF"}), 400

    PDF_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    file.save(PDF_UPLOAD_DIR / filename)
    return jsonify({"saved": filename})


@app.delete("/pdfs/<path:filename>")
def delete_pdf(filename):
    _bump_activity()
    safe_name = secure_filename(filename)
    target = PDF_UPLOAD_DIR / safe_name
    if safe_name and target.is_file() and target.suffix.lower() == ".pdf":
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
  ul#pdf-list { list-style: none; padding: 0; }
  ul#pdf-list li { display: flex; align-items: center; justify-content: space-between;
                    padding: 0.4rem 0.6rem; border-bottom: 1px solid #ddd; }
  ul#pdf-list li button { font-size: 0.85rem; padding: 0.2rem 0.6rem; }
  #upload-msg { margin-top: 0.5rem; font-size: 0.9rem; }
</style>
</head>
<body>
  <h1>Ingest</h1>

  <h2>PDFs</h2>
  <p>Uploaded PDFs are converted to markdown via <code>marker</code> and included in the next ingest run.</p>
  <input type="file" id="pdf-file" accept="application/pdf">
  <button onclick="uploadPdf()">Upload</button>
  <div id="upload-msg"></div>
  <ul id="pdf-list"></ul>

  <h2>Run</h2>
  <p>Incremental by default: only new/changed files are re-chunked and re-embedded, unchanged
  files (and unchanged PDFs) are skipped, removed files are cleaned up from Qdrant.</p>
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
      </tr>
    </thead>
    <tbody></tbody>
  </table>

<script>
const btn = document.getElementById('run-btn');
const statusEl = document.getElementById('status');
const logEl = document.getElementById('log');
const pdfFileEl = document.getElementById('pdf-file');
const pdfListEl = document.getElementById('pdf-list');
const uploadMsgEl = document.getElementById('upload-msg');
const forceFullEl = document.getElementById('force-full');

async function loadPdfs() {
  const res = await fetch('/pdfs');
  const names = await res.json();
  pdfListEl.innerHTML = '';
  for (const name of names) {
    const li = document.createElement('li');
    const label = document.createElement('span');
    label.textContent = name;
    const delBtn = document.createElement('button');
    delBtn.textContent = 'Delete';
    delBtn.onclick = () => deletePdf(name);
    li.appendChild(label);
    li.appendChild(delBtn);
    pdfListEl.appendChild(li);
  }
}

async function uploadPdf() {
  const file = pdfFileEl.files[0];
  if (!file) {
    uploadMsgEl.textContent = 'Please select a PDF file first.';
    return;
  }
  const formData = new FormData();
  formData.append('file', file);
  const res = await fetch('/pdfs', { method: 'POST', body: formData });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    uploadMsgEl.textContent = 'Error: ' + (body.error || res.statusText);
    return;
  }
  uploadMsgEl.textContent = body.saved + ' uploaded.';
  pdfFileEl.value = '';
  loadPdfs();
}

async function deletePdf(name) {
  await fetch('/pdfs/' + encodeURIComponent(name), { method: 'DELETE' });
  loadPdfs();
}

loadPdfs();

async function loadSources() {
  const tbody = document.querySelector('#sources-table tbody');
  const summary = document.getElementById('sources-summary');
  try {
    const res = await fetch('/sources');
    const data = await res.json();
    if (!data.indexed) {
      summary.textContent = 'Nothing indexed yet. Run an ingest first.';
      tbody.innerHTML = '';
      return;
    }
    summary.textContent = `Collection: ${data.collection || '-'} | dim=${data.embedding_dim || '-'} | ${data.total_sources} sources | ${data.total_chunks} chunks`;
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
      tr.appendChild(tdSrc);
      tr.appendChild(tdChunks);
      tbody.appendChild(tr);
    }
  } catch (e) {
    summary.textContent = 'Failed to load sources: ' + e.message;
  }
}

loadSources();

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
