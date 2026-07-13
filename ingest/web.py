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

from flask import Flask, Response, jsonify

from ingest.ingest import main as run_ingest

app = Flask(__name__)

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


_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Ingest</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 760px; margin: 2rem auto; padding: 0 1rem; }
  button { font-size: 1rem; padding: 0.5rem 1rem; cursor: pointer; }
  button:disabled { cursor: not-allowed; opacity: 0.6; }
  #status { margin: 1rem 0; font-weight: 600; }
  .idle { color: #666; }
  .running { color: #b8860b; }
  .done { color: #2a7a2a; }
  .error { color: #c0392b; }
  pre { background: #111; color: #ddd; padding: 1rem; height: 420px; overflow-y: auto;
        white-space: pre-wrap; word-break: break-word; border-radius: 6px; }
</style>
</head>
<body>
  <h1>Ingest</h1>
  <p>Re-indexes the notes repo into Qdrant. This deletes and rebuilds the collection.</p>
  <button id="run-btn" onclick="runIngest()">Ingest starten</button>
  <div id="status" class="idle">idle</div>
  <pre id="log"></pre>

<script>
const btn = document.getElementById('run-btn');
const statusEl = document.getElementById('status');
const logEl = document.getElementById('log');

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
