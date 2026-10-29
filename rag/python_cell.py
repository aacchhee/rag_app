"""
Optional runnable Python cell attached to a generated exercise.

After the exercise text has been streamed, a second LLM call offers the model
the create_python_cell tool. The model decides whether Python fits the
exercise (or is forced to when the student asks for "more computational").
The starter code is validated and lightly repaired before it reaches the
browser, where it runs in Pyodide.
"""

import ast
import re

ALLOWED_PACKAGES = ("numpy", "sympy", "matplotlib")

PYTHON_CELL_TOOL = {
    "type": "function",
    "function": {
        "name": "create_python_cell",
        "description": (
            "Decide whether the exercise gets a runnable Python cell and, if so, provide its "
            "starter code. The student edits and runs it in the browser (Pyodide)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "use_python": {
                    "type": "boolean",
                    "description": "true if computing, iterating, simulating or plotting is a natural part of solving this exercise.",
                },
                "starter_code": {
                    "type": "string",
                    "description": (
                        "Valid, runnable Python 3 starter code. Mark the parts the student "
                        "must write with '# TODO' comments and give every function or loop "
                        "a body (use 'pass' or '...'). Must NOT contain the solution."
                    ),
                },
                "packages": {
                    "type": "array",
                    "items": {"type": "string", "enum": list(ALLOWED_PACKAGES)},
                    "description": "Packages the starter code imports.",
                },
            },
            # Deciding first, explicitly, is far more consistent across models than
            # leaving the call itself optional (tool_choice "auto")
            "required": ["use_python"],
        },
    },
}

# "expected an indented block after function definition on line 4"
_EMPTY_BLOCK_RE = re.compile(r"expected an indented block after .* on line (\d+)")
_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*\n|\n?```\s*$")


def repair_starter_code(code: str, max_fixes: int = 20) -> str | None:
    """
    Return code that parses as Python, or None if it can't be repaired.

    Models often leave a block containing only '# TODO' comments, which is a
    syntax error; the fix is to give such blocks a 'pass' body.
    """
    code = _FENCE_RE.sub("", (code or "").strip()).strip("\n")
    if not code.strip():
        return None

    lines = code.split("\n")
    for _ in range(max_fixes + 1):
        try:
            ast.parse("\n".join(lines))
            return "\n".join(lines) + "\n"
        except SyntaxError as e:
            m = _EMPTY_BLOCK_RE.search(e.msg or "")
            if not m:
                return None
            header_idx = int(m.group(1)) - 1
            if not 0 <= header_idx < len(lines):
                return None
            # Place 'pass' after the comments that belong to the empty block
            indent = len(lines[header_idx]) - len(lines[header_idx].lstrip()) + 4
            insert_at = header_idx + 1
            while insert_at < len(lines) and lines[insert_at].strip().startswith("#"):
                insert_at += 1
            lines.insert(insert_at, " " * indent + "pass")
    return None


MAX_CONTEXT_CODE_CHARS = 6000
MAX_CONTEXT_OUTPUT_CHARS = 3000


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "\n… (truncated)"


def student_code_context(data: dict) -> str:
    """
    Prompt section describing the student's work in the exercise's Python cell,
    or "" when the request carries no code. Sent by the frontend as
    student_code plus code_run = {status, output, plots}, where status is
    "done", "error", "stopped", "stale" (edited since the last run) or "never".
    """
    code = data.get("student_code")
    if not isinstance(code, str) or not code.strip():
        return ""
    run = data.get("code_run") if isinstance(data.get("code_run"), dict) else {}
    status = run.get("status") or "never"
    output = run.get("output") if isinstance(run.get("output"), str) else ""
    plots = run.get("plots") if isinstance(run.get("plots"), int) else 0

    status_text = {
        "done": "ran to completion",
        "error": "raised an error",
        "stopped": "was stopped by the student before finishing (possibly an endless loop)",
        "stale": "has been edited since it was last run, so the output below may be outdated",
        "never": "has not been run yet",
    }.get(status, "has not been run yet")

    parts = [
        "STUDENT'S PYTHON CODE (written in the exercise's Python cell):",
        "```python\n" + _clip(code.rstrip(), MAX_CONTEXT_CODE_CHARS) + "\n```",
        "The last run " + status_text + ".",
    ]
    if status != "never":
        parts.append("OUTPUT OF THE LAST RUN:\n```\n" + (_clip(output.rstrip(), MAX_CONTEXT_OUTPUT_CHARS) or "(no output)") + "\n```")
    if plots:
        parts.append(f"The code produced {plots} plot(s); you cannot see them, only that they exist.")
    parts.append(
        "Take the code and its output into account together with any written answer. "
        "Refer to specific lines when that helps, but do not hand over finished code for the TODO parts."
    )
    return "\n\n".join(parts)


def build_python_cell(args: dict, *, force: bool = False) -> dict | None:
    """
    Validate tool-call arguments into the cell sent to the browser.
    force=True (student asked for more computational) ignores use_python=false
    as long as starter code was provided.
    """
    if not force and args.get("use_python") is False:
        return None
    code = repair_starter_code(args.get("starter_code") or "")
    if code is None:
        return None
    packages = [p for p in (args.get("packages") or []) if p in ALLOWED_PACKAGES]
    # Trust the imports over the declared list so Pyodide preloads what is used
    for pkg in ALLOWED_PACKAGES:
        if re.search(rf"^\s*(import|from)\s+{pkg}\b", code, re.M) and pkg not in packages:
            packages.append(pkg)
    return {"starter_code": code, "packages": packages}
