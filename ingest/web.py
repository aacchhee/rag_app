"""
Minimal web UI for the ingest service.

Wraps ingest.ingest.main() so a re-index can be triggered from a browser
instead of a shell, with live status/log output. Runs as a small Flask app;
the actual ingest logic in ingest/ingest.py is untouched.
"""

import contextlib
import io
import threading
import traceback
from datetime import datetime, timezone

from flask import Flask, Response, jsonify, request
from werkzeug.utils import secure_filename

from ingest.ingest import PDF_UPLOAD_DIR, main as run_ingest

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB per upload

_lock = threading.Lock()
_state = {
    "status": "idle",  # idle | running | done | error
    "started_at": None,
    "finished_at": None,
    "error": None,
    "log": "",
}


class _LogStream(io.StringIO):
    def write(self, s):
        with _lock:
            _state["log"] += s
        return super().write(s)


def _run_job():
    with _lock:
        _state.update(
            status="running",
            started_at=datetime.now(timezone.utc).isoformat(),
            finished_at=None,
            error=None,
            log="",
        )

    try:
        with contextlib.redirect_stdout(_LogStream()):
            run_ingest()
        with _lock:
            _state["status"] = "done"
    except Exception:
        with _lock:
            _state["status"] = "error"
            _state["error"] = traceback.format_exc()
    finally:
        with _lock:
            _state["finished_at"] = datetime.now(timezone.utc).isoformat()


@app.get("/")
def index():
    return Response(_PAGE, mimetype="text/html")


@app.get("/status")
def status():
    with _lock:
        return jsonify(dict(_state))


@app.post("/run")
def trigger():
    with _lock:
        if _state["status"] == "running":
            return jsonify({"error": "ingest already running"}), 409
    threading.Thread(target=_run_job, daemon=True).start()
    return jsonify({"started": True})


@app.get("/pdfs")
def list_pdfs():
    PDF_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    return jsonify(sorted(p.name for p in PDF_UPLOAD_DIR.glob("*.pdf")))


@app.post("/pdfs")
def upload_pdf():
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
  <p>Re-indexes the notes repo into Qdrant. This deletes and rebuilds the collection.</p>
  <button id="run-btn" onclick="runIngest()">Start ingest</button>
  <div id="status" class="idle">idle</div>
  <pre id="log"></pre>

<script>
const btn = document.getElementById('run-btn');
const statusEl = document.getElementById('status');
const logEl = document.getElementById('log');
const pdfFileEl = document.getElementById('pdf-file');
const pdfListEl = document.getElementById('pdf-list');
const uploadMsgEl = document.getElementById('upload-msg');

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

async function runIngest() {
  const res = await fetch('/run', { method: 'POST' });
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
