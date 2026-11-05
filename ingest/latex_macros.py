"""
Expand author-defined LaTeX macros into LaTeX that MathJax understands.

Math sources written in LaTeX (PreTeXt books, .tex notes) define their own
shorthands: \\R, \\inv, \\Span, \\smallmat... MathJax and the LLM know none of
them, so we expand them at conversion time, the way TeX would:

    ms = MacroSet()
    ms.load(r"\\def\\R{\\mathbf R} \\newcommand{\\inv}{^{-1}}")
    ms.expand(r"A\\inv \\in \\R^n")   # -> 'A^{-1} \\in \\mathbf R^n'

Understood: \\def, \\newcommand (and re-/provide-), \\DeclareMathOperator, \\let,
\\newenvironment, \\definecolor, and \\usepackage of local .sty files. Definitions
met inside the expanded text (a \\def inside one equation) apply to the rest of
that text only. The spalign package (\\spalignmat{1 2; 3 4} and friends, used by
ILA as \\mat, \\vec, \\amat, \\syseq) is implemented natively, as its TeX
internals are beyond this expander.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Macro:
    nargs: int = 0
    default: str | None = None  # default of an optional first argument (\newcommand{\x}[2][d])
    body: str = ""


@dataclass
class Alias:
    """\\let\\a=\\b where \\b has no definition here: \\a stands for that command."""
    target: str


_CTRL_WORD_END_RE = re.compile(r"\\[A-Za-z@]+$")
_DIMEN_RE = re.compile(r"\s*=?\s*-?[\d.]+\s*(pt|em|ex|mm|cm|in|mu|bp|pc|sp)")

_DEFINERS = frozenset({
    "def", "gdef", "edef", "xdef", "newcommand", "renewcommand", "providecommand",
    "DeclareRobustCommand", "DeclareMathOperator", "let", "newenvironment",
    "renewenvironment", "definecolor", "usepackage", "RequirePackage",
})

# Packages implemented natively below: never parse their .sty files
_NATIVE_PACKAGES = frozenset({"spalign"})


class MacroSet:
    def __init__(self, search_paths: list[Path] | None = None):
        self.macros: dict[str, Macro | Alias] = {}
        self.envs: dict[str, tuple[Macro, str]] = {}  # name -> (begin code with its args, end code)
        self.colors: dict[str, tuple[str, str]] = {}  # name -> (model, spec)
        self.delims = ("(", ")")  # spalign: matrices and vectors
        self.sysdelims = (r"\{", ".")  # spalign: systems of equations
        self.search_paths = list(search_paths or [])
        self._loaded: set[str] = set()
        self.missing_packages: set[str] = set()

    def copy(self) -> "MacroSet":
        ms = MacroSet(self.search_paths)
        ms.macros, ms.envs, ms.colors = dict(self.macros), dict(self.envs), dict(self.colors)
        ms.delims, ms.sysdelims = self.delims, self.sysdelims
        ms._loaded, ms.missing_packages = self._loaded, self.missing_packages
        return ms

    # --- loading definitions ---------------------------------------------

    def load(self, text: str, at_letter: bool = False) -> None:
        """Read the definitions in `text` (a preamble or .sty file); ignore everything else."""
        text = _strip_comments(text)
        i = 0
        while True:
            i = text.find("\\", i)
            if i == -1:
                return
            name, j = _read_cs(text, i, at_letter)
            if name in _DEFINERS or name in ("spaligndelims", "spalignsysdelims"):
                end = self._define(name, text, j, at_letter)
                i = end if end is not None else j
            else:
                i = j

    def load_file(self, path: Path) -> None:
        self.load(path.read_text(encoding="utf-8", errors="replace"), at_letter=path.suffix in (".sty", ".cls"))

    def _use_package(self, name: str) -> None:
        name = name.strip()
        if not name or name in _NATIVE_PACKAGES or name in self._loaded:
            return
        self._loaded.add(name)
        for base in self.search_paths:
            for cand in (base / f"{name}.sty", *base.rglob(f"{name}.sty")):
                if cand.is_file():
                    self.load_file(cand)
                    return
        self.missing_packages.add(name)  # a standard package (amsmath, tikz...): nothing to define

    def _define(self, name: str, s: str, j: int, at_letter: bool = False) -> int | None:
        """Parse the definition starting after \\<name> at s[j:]. Returns its end, or None."""
        if name in ("def", "gdef", "edef", "xdef"):
            j = _skip_ws(s, j)
            if j >= len(s) or s[j] != "\\":
                return None
            target, j = _read_cs(s, j, at_letter)
            m = re.match(r"(?:#\d)*", s[j:])
            params = m.group(0)
            j += len(params)
            if j >= len(s) or s[j] != "{":
                return None  # delimited parameters (\def\x#1.{...}): not supported
            body, j = _read_group(s, j)
            if name in ("edef", "xdef"):
                body = self.expand(body)  # \edef expands its body when defined
            self.macros[target] = Macro(len(params) // 2, None, body)
            return j

        if name in ("newcommand", "renewcommand", "providecommand", "DeclareRobustCommand"):
            j = _skip_star(s, j)
            target, j = _read_target(s, j, at_letter)
            if target is None:
                return None
            n, j = _read_optional(s, j)
            default, j = _read_optional(s, j)
            j = _skip_ws(s, j)
            if j >= len(s) or s[j] != "{":
                return None
            body, j = _read_group(s, j)
            if name != "providecommand" or target not in self.macros:
                self.macros[target] = Macro(int(n) if n and n.strip().isdigit() else 0, default, body)
            return j

        if name == "DeclareMathOperator":
            star = j < len(s) and s[j] == "*"
            j = _skip_star(s, j)
            target, j = _read_target(s, j, at_letter)
            arg = _read_arg(s, j)
            if target is None or arg is None:
                return None
            text, j = arg
            self.macros[target] = Macro(0, None, rf"\operatorname{'*' if star else ''}{{{text}}}")
            return j

        if name == "let":
            j = _skip_ws(s, j)
            if j >= len(s) or s[j] != "\\":
                return None
            target, j = _read_cs(s, j, at_letter)
            j = _skip_ws(s, j)
            if j < len(s) and s[j] == "=":
                j = _skip_ws(s, j + 1)
            if j >= len(s):
                return None
            if s[j] == "\\":
                source, j = _read_cs(s, j, at_letter)
                self.macros[target] = self.macros.get(source, Alias(source))
            else:
                self.macros[target] = Macro(0, None, s[j])
                j += 1
            return j

        if name in ("newenvironment", "renewenvironment"):
            j = _skip_star(s, j)
            arg = _read_arg(s, j)
            if arg is None:
                return None
            env, j = arg
            n, j = _read_optional(s, j)
            default, j = _read_optional(s, j)
            begin, end = _read_arg(s, j), None
            if begin is not None:
                end = _read_arg(s, begin[1])
            if begin is None or end is None:
                return None
            self.envs[env.strip()] = (Macro(int(n) if n and n.strip().isdigit() else 0, default, begin[0]), end[0])
            return end[1]

        if name == "definecolor":
            args, k = [], j
            for _ in range(3):
                arg = _read_arg(s, k)
                if arg is None:
                    return None
                args.append(arg[0].strip())
                k = arg[1]
            self.colors[args[0]] = (args[1], args[2])
            return k

        if name in ("usepackage", "RequirePackage"):
            _, j = _read_optional(s, j)
            arg = _read_arg(s, j)
            if arg is None:
                return None
            for pkg in arg[0].split(","):
                self._use_package(pkg)
            return arg[1]

        if name in ("spaligndelims", "spalignsysdelims"):
            a = _read_arg(s, j)
            b = _read_arg(s, a[1]) if a else None
            if b is None:
                return None
            if name == "spaligndelims":
                self.delims = (a[0], b[0])
            else:
                self.sysdelims = (a[0], b[0])
            return b[1]

        return None

    # --- expansion ---------------------------------------------------------

    def expand(self, text: str, keep_definitions: bool = False, max_steps: int = 20000) -> str:
        """
        `text` with every known macro expanded, recursively. Definitions made
        inside `text` stay local to it, unless `keep_definitions`: then they hold
        for later calls too, as a \\def in one formula does on a MathJax page.
        """
        ms = self.copy()
        try:
            return ms._expand(text, max_steps)
        finally:
            if keep_definitions:
                self.macros, self.envs, self.colors = ms.macros, ms.envs, ms.colors

    def _expand(self, text: str, max_steps: int) -> str:
        ms = self
        out: list[str] = []
        s, i = text, 0
        while i < len(s):
            k = s.find("\\", i)
            if k == -1:
                out.append(s[i:])
                break
            out.append(s[i:k])
            name, j = _read_cs(s, k)
            max_steps -= 1
            result = ms._expand_one(name, s, j) if max_steps > 0 and name else None
            if result is None:
                out.append(s[k:j])
                i = j
                continue
            repl, end, rescan = result
            if _CTRL_WORD_END_RE.search(repl) and end < len(s) and s[end].isalpha():
                repl += " "  # keep \cmd from merging with following letters
            if rescan:
                s, i = repl + s[end:], 0
            else:
                out.append(repl)
                s, i = s[end:], 0
        return "".join(out)

    def _expand_one(self, name: str, s: str, j: int):
        """(replacement, end, rescan) for the command \\<name> ending at s[j], or None to keep it."""
        if name in _DEFINERS:
            end = self._define(name, s, j)
            return ("", end, False) if end is not None else None
        if name in ("begin", "end"):
            arg = _read_arg(s, j)
            if arg is not None and name == "begin" and arg[0].strip() in _ENV_BUILTINS:
                return _ENV_BUILTINS[arg[0].strip()](self, s, arg[1])
            if arg is None or arg[0].strip() not in self.envs:
                return None
            begin, end_code = self.envs[arg[0].strip()]
            if name == "end":
                return end_code, arg[1], True
            return self._apply(begin, s, arg[1])

        definition = self.macros.get(name)
        if isinstance(definition, Macro):
            return self._apply(definition, s, j)
        if isinstance(definition, Alias):
            if definition.target in _BUILTINS:
                return _BUILTINS[definition.target](self, s, j)
            return "\\" + definition.target, j, False
        if name in _BUILTINS:
            return _BUILTINS[name](self, s, j)
        return None

    def _apply(self, macro: Macro, s: str, j: int):
        args: list[str] = []
        n = macro.nargs
        if macro.default is not None:
            opt, j = _read_optional(s, j)
            args.append(macro.default if opt is None else opt)
            n -= 1
        for _ in range(n):
            arg = _read_arg(s, j)
            if arg is None:
                return None
            args.append(arg[0])
            j = arg[1]
        return _substitute(macro.body, args), j, True


# --- tokenizing ----------------------------------------------------------

def _strip_comments(text: str) -> str:
    return re.sub(r"(?<!\\)%[^\n]*", "", text)


def _read_cs(s: str, i: int, at_letter: bool = False) -> tuple[str, int]:
    """The control sequence at s[i] == '\\': (name, end). A control symbol (\\,) has a non-letter name."""
    j = i + 1
    if j >= len(s):
        return "", j
    if s[j].isascii() and (s[j].isalpha() or (at_letter and s[j] == "@")):
        while j < len(s) and s[j].isascii() and (s[j].isalpha() or (at_letter and s[j] == "@")):
            j += 1
        return s[i + 1:j], j
    return s[j], j + 1


def _skip_ws(s: str, j: int) -> int:
    while j < len(s) and s[j].isspace():
        j += 1
    return j


def _skip_star(s: str, j: int) -> int:
    j = _skip_ws(s, j)
    return j + 1 if j < len(s) and s[j] == "*" else j


def _read_group(s: str, j: int) -> tuple[str, int]:
    """The balanced group at s[j] == '{': (contents, end)."""
    depth, k = 0, j
    while k < len(s):
        c = s[k]
        if c == "\\":
            k += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[j + 1:k], k + 1
        k += 1
    return s[j + 1:], len(s)


def _read_arg(s: str, j: int) -> tuple[str, int] | None:
    """One macro argument: a {group}, a control sequence or a single character."""
    j = _skip_ws(s, j)
    if j >= len(s) or s[j] == "}":
        return None
    if s[j] == "{":
        return _read_group(s, j)
    if s[j] == "\\":
        _, k = _read_cs(s, j)
        return s[j:k], k
    return s[j], j + 1


def _read_optional(s: str, j: int) -> tuple[str | None, int]:
    k = _skip_ws(s, j)
    if k >= len(s) or s[k] != "[":
        return None, j
    depth, m = 0, k + 1
    while m < len(s):
        c = s[m]
        if c == "\\":
            m += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        elif c == "]" and depth == 0:
            return s[k + 1:m], m + 1
        m += 1
    return None, j


def _read_target(s: str, j: int, at_letter: bool) -> tuple[str | None, int]:
    """The command being defined: \\name or {\\name}."""
    j = _skip_ws(s, j)
    if j < len(s) and s[j] == "{":
        inner, end = _read_group(s, j)
        inner = inner.strip()
        if inner.startswith("\\"):
            return _read_cs(inner, 0, at_letter)[0], end
        return None, end
    if j < len(s) and s[j] == "\\":
        return _read_cs(s, j, at_letter)
    return None, j


def _substitute(body: str, args: list[str]) -> str:
    out: list[str] = []
    i = 0
    while i < len(body):
        c = body[i]
        if c == "#" and i + 1 < len(body):
            nxt = body[i + 1]
            if nxt == "#":
                out.append("#")
                i += 2
                continue
            if nxt.isdigit() and 0 < int(nxt) <= len(args):
                arg = args[int(nxt) - 1]
                if arg[:1].isalpha() and _CTRL_WORD_END_RE.search("".join(out[-64:])):  # \int#1 with x
                    out.append(" ")
                out.append(arg)
                i += 2
                continue
        if c == "\\" and i + 1 < len(body):
            out.append(body[i:i + 2])
            i += 2
            continue
        out.append(c)
        i += 1
    return "".join(out)


# --- spalign -------------------------------------------------------------
# \spalignmat{1 2; 3 4}: cells split on spaces or commas, rows on semicolons.
# As in TeX, a space right after a control word (\lambda 0) does not split.

def _split_top(s: str, sep: str) -> list[str]:
    parts, depth, cur, i = [], 0, [], 0
    while i < len(s):
        c = s[i]
        if c == "\\":
            _, j = _read_cs(s, i)
            cur.append(s[i:j])
            i = j
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        if c == sep and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(c)
        i += 1
    parts.append("".join(cur))
    return parts


def _split_cells(row: str) -> list[str]:
    cells, cur, depth, i = [], "", 0, 0
    while i < len(row):
        c = row[i]
        if c == "\\":
            _, j = _read_cs(row, i)
            cur += row[i:j]
            i = j
            continue
        if depth == 0 and (c.isspace() or c == ","):
            if c.isspace() and _CTRL_WORD_END_RE.search(cur):
                cur += " "  # TeX skips spaces after a control word: no new cell
                i += 1
                continue
            if cur:
                cells.append(cur)
            cur = ""
            i += 1
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        cur += c
        i += 1
    if cur:
        cells.append(cur)
    return cells


def _spalign_rows(content: str) -> list[list[str]]:
    rows = [_split_cells(r) for r in _split_top(content, ";")]
    return [r for r in rows if r]


_MATRIX_ENVS = {("(", ")"): "pmatrix", ("[", "]"): "bmatrix", ("|", "|"): "vmatrix",
                (r"\{", r"\}"): "Bmatrix", ("\\|", "\\|"): "Vmatrix", (".", "."): "matrix"}


def _matrix(ms: MacroSet, rows: list[list[str]], colspec: str, star: bool) -> str:
    body = r" \\ ".join(" & ".join(r) for r in rows)
    delims = (".", ".") if star else ms.delims
    if set(colspec) == {"c"} and delims in _MATRIX_ENVS:
        env = _MATRIX_ENVS[delims]
        return rf"\begin{{{env}}} {body} \end{{{env}}}"
    array = rf"\begin{{array}}{{{colspec}}} {body} \end{{array}}"
    if delims == (".", "."):
        return array
    return rf"\left{delims[0]} {array} \right{delims[1]}"


def _spalign_args(s: str, j: int, default_align: str):
    star = j < len(s) and s[j] == "*"
    if star:
        j += 1
    align, j = _read_optional(s, j)
    arg = _read_arg(s, j)
    if arg is None:
        return None
    return star, (align or default_align).strip() or default_align, _spalign_rows(arg[0]), arg[1]


def _sp_mat(ms, s, j):
    a = _spalign_args(s, j, "c")
    if a is None:
        return None
    star, align, rows, end = a
    n = max((len(r) for r in rows), default=1)
    return _matrix(ms, rows, align * n, star), end, True


def _sp_augmented(split):
    def handler(ms, s, j):
        a = _spalign_args(s, j, "r")
        if a is None:
            return None
        star, align, rows, end = a
        n = max((len(r) for r in rows), default=1)
        left = split(n)
        return _matrix(ms, rows, align * left + "|" + align * (n - left), star), end, True
    return handler


def _sp_vec(ms, s, j):
    a = _spalign_args(s, j, "c")
    if a is None:
        return None
    star, align, rows, end = a
    return _matrix(ms, [[c] for r in rows for c in r], align, star), end, True


def _sp_sys(ms, s, j):
    star = j < len(s) and s[j] == "*"
    arg = _read_arg(s, j + 1 if star else j)
    if arg is None:
        return None
    rows = _spalign_rows(arg[0])
    cols = max((len(r) for r in rows), default=1)
    lines = []
    for r in rows:
        cells = []
        for k, c in enumerate(r):
            c = "" if c == r"\." else c.replace(r"\+", r"\mathbin{\phantom{+}}").replace(r"\=", r"\mathrel{\phantom{=}}")
            cells.append(c if k % 2 == 0 else "{}" + c + "{}")  # odd columns hold + and =
        lines.append(" & ".join(cells))
    pairs = (cols + 1) // 2
    body = rf"\begin{{alignedat}}{{{pairs}}} " + r" \\ ".join(lines) + r" \end{alignedat}"
    left, right = (".", ".") if star else ms.sysdelims
    if (left, right) != (".", "."):
        body = rf"\left{left} {body} \right{right}"
    return body, arg[1], True


def _sp_delims(attr):
    def handler(ms, s, j):
        a = _read_arg(s, j)
        b = _read_arg(s, a[1]) if a else None
        if b is None:
            return None
        setattr(ms, attr, (a[0], b[0]))  # for the rest of this text
        return "", b[1], False
    return handler


def _sp_tabspace(ms, s, j):
    m = _DIMEN_RE.match(s, j)
    return ("", m.end(), False) if m else None


# --- other built-ins -----------------------------------------------------

def _ensuremath(ms, s, j):
    arg = _read_arg(s, j)
    return (arg[0], arg[1], True) if arg else None


# xcolor's base colors, as rgb
_BASE_COLORS = {
    "red": (1, 0, 0), "green": (0, 1, 0), "blue": (0, 0, 1), "cyan": (0, 1, 1), "magenta": (1, 0, 1),
    "yellow": (1, 1, 0), "black": (0, 0, 0), "white": (1, 1, 1), "gray": (.5, .5, .5),
    "darkgray": (.25, .25, .25), "lightgray": (.75, .75, .75), "brown": (.75, .5, .25),
    "lime": (.75, 1, 0), "olive": (.5, .5, 0), "orange": (1, .5, 0), "pink": (1, .75, .75),
    "purple": (.75, 0, .25), "teal": (0, .5, .5), "violet": (.5, 0, .5),
}


def _to_rgb(model: str, spec: str) -> tuple[float, float, float] | None:
    try:
        if model == "rgb":
            r, g, b = (float(x) for x in spec.split(","))
            return r, g, b
        if model == "RGB":
            r, g, b = (float(x) / 255 for x in spec.split(","))
            return r, g, b
        if model == "HTML":
            return tuple(int(spec[i:i + 2], 16) / 255 for i in (0, 2, 4))  # type: ignore[return-value]
        if model == "gray":
            return (float(spec),) * 3  # type: ignore[return-value]
    except ValueError:
        pass
    return None


def _resolve_color(ms: MacroSet, name: str) -> tuple[float, float, float] | None:
    """rgb of a color name or xcolor expression (green!70!black), if we know it."""
    def one(n: str):
        n = n.strip()
        if n in ms.colors:
            return _to_rgb(*ms.colors[n])
        return _BASE_COLORS.get(n.replace(" ", "").lower())
    parts = name.split("!")
    rgb = one(parts[0])
    k = 1
    while rgb is not None and k < len(parts):
        try:
            p = float(parts[k]) / 100
        except ValueError:
            return None
        other = one(parts[k + 1]) if k + 1 < len(parts) else (1, 1, 1)  # a!p alone mixes with white
        if other is None:
            return None
        rgb = tuple(p * a + (1 - p) * b for a, b in zip(rgb, other))  # type: ignore[assignment]
        k += 2
    return rgb


def _color(cmd):
    """
    \\color{name}: a color MathJax can't name (one the source defined, an xcolor
    mix like green!70!black, a color model MathJax lacks) becomes its rgb value.
    """
    def handler(ms, s, j):
        k = _skip_ws(s, j)
        model, k = _read_optional(s, k)
        arg = _read_arg(s, k)
        if arg is None:
            return None
        spec = arg[0].strip()
        if model is not None:
            if model.strip() in ("rgb", "RGB", "gray"):
                return None
            rgb = _to_rgb(model.strip(), spec)
        elif spec in ms.colors or "!" in spec or " " in spec:
            rgb = _resolve_color(ms, spec)
        else:
            return None  # a plain name (red, blue): MathJax knows it
        if rgb is None:
            return None
        value = ",".join(f"{x:.4g}" for x in rgb)
        return rf"\{cmd}[rgb]{{{value}}}", arg[1], False
    return handler


def _expandafter(ms, s, j):
    """\\expandafter A B: expand B once, then put A in front of the result."""
    first = _read_token(s, j)
    if first is None:
        return None
    a, k = first
    if k < len(s) and s[k] == "\\":
        name, m = _read_cs(s, k)
        r = ms._expand_one(name, s, m)
        if r is not None:
            repl, end, _ = r
            return a + repl + s[end:], len(s), True
    return a, k, True  # B is not expandable: \expandafter does nothing


def _read_token(s: str, j: int) -> tuple[str, int] | None:
    if j >= len(s):
        return None
    if s[j] == "\\":
        _, k = _read_cs(s, j)
        return s[j:k], k
    return s[j], j + 1


# TeX layout that MathJax lacks, mostly in arrow labels: keep the text, drop the layout

_LAYOUT_RE = re.compile(r"\\(?:tiny|scriptsize|footnotesize|small|centering|raggedright|raggedleft|noindent)(?![A-Za-z])\s*")


def _text_lines(content: str) -> str:
    lines = [ln.strip() for ln in _LAYOUT_RE.sub("", content).split(r"\\")]
    lines = [rf"\text{{{ln}}}" for ln in lines if ln]
    if len(lines) == 1:
        return lines[0]
    return r"\begin{array}{c} " + r" \\ ".join(lines) + r" \end{array}"


def _parbox(ms, s, j):
    _, j = _read_optional(s, j)
    width = _read_arg(s, j)
    content = _read_arg(s, width[1]) if width else None
    return (_text_lines(content[0]), content[1], True) if content else None


def _makebox(ms, s, j):
    _, j = _read_optional(s, j)
    _, j = _read_optional(s, j)
    content = _read_arg(s, j)
    return (rf"\text{{{content[0]}}}", content[1], True) if content else None


def _hbox(ms, s, j):
    m = re.match(r"\s*(?:to|spread)\s*-?[\d.]+\s*(pt|em|ex|mm|cm|in|bp)", s[j:])
    return (r"\hbox", j + m.end(), False) if m else None  # \hbox to 2cm{...} -> \hbox{...}


def _drop(ms, s, j):
    return "", j, False


def _unbox(ms, s, j):
    arg = _read_arg(s, j)
    return (arg[0], arg[1], True) if arg else None


def _env_body(s: str, j: int, env: str) -> tuple[str, int]:
    """Text up to the matching \\end{env}, and the position after it."""
    depth, k = 1, j
    pat = re.compile(r"\\(begin|end)\s*\{" + re.escape(env) + r"\}")
    for m in pat.finditer(s, j):
        depth += 1 if m.group(1) == "begin" else -1
        if depth == 0:
            return s[j:m.start()], m.end()
    return s[j:], len(s)


def _minipage(ms, s, j):
    _, j = _read_optional(s, j)
    width = _read_arg(s, j)
    if width is None:
        return None
    body, end = _env_body(s, width[1], "minipage")
    return _text_lines(body), end, True


def _picture(env):
    def handler(ms, s, j):
        _, end = _env_body(s, j, env)
        return r"\text{[diagram]}", end, False  # MathJax can't draw it
    return handler


_ENV_BUILTINS = {
    "minipage": _minipage,
    "tikzpicture": _picture("tikzpicture"),
    "tikzcd": _picture("tikzcd"),
    "pspicture": _picture("pspicture"),
}


_BUILTINS = {
    "expandafter": _expandafter,
    "parbox": _parbox,
    "makebox": _makebox,
    "hbox": _hbox,
    "vbox": _unbox,
    "hss": _drop,
    "centering": _drop,
    "noindent": _drop,
    "ensuremath": _ensuremath,
    "color": _color("color"),
    "textcolor": _color("textcolor"),
    "spalignmat": _sp_mat,
    "spalignaugmat": _sp_augmented(lambda n: n - 1),
    "spalignaugmathalf": _sp_augmented(lambda n: n // 2),
    "spalignvector": _sp_vec,
    "spalignsys": _sp_sys,
    "spaligndelims": _sp_delims("delims"),
    "spalignsysdelims": _sp_delims("sysdelims"),
    "spalignsystabspace": _sp_tabspace,
}


_TEXT_CMD_RE = re.compile(r"\\(text|mbox|hbox|textrm|textnormal)\s*\{")


def unnest_text_math(tex: str) -> str:
    """
    \\text{$n\\times n$ matrices} -> n\\times n\\text{ matrices}. Math inside \\text
    is valid TeX, but a '$' in a $-delimited formula ends it early in Markdown.
    """
    out, i = [], 0
    for m in _TEXT_CMD_RE.finditer(tex):
        if m.start() < i:
            continue
        content, end = _read_group(tex, m.end() - 1)
        if "$" not in content:
            continue
        out.append(tex[i:m.start()])
        for k, piece in enumerate(re.split(r"(?<!\\)\$", content)):
            if k % 2:
                out.append("{" + piece + "}")
            elif piece:
                out.append(rf"\{m.group(1)}{{{piece}}}")
        i = end
    out.append(tex[i:])
    return "".join(out)


# --- report --------------------------------------------------------------
# Commands MathJax (with its autoloaded extensions) renders. Anything else left
# in converted math probably still needs a definition. Not exhaustive; a
# command listed by the report may still be fine.

MATHJAX_COMMANDS = frozenset("""
alpha beta gamma delta epsilon varepsilon zeta eta theta vartheta iota kappa varkappa lambda mu nu xi omicron
pi varpi rho varrho sigma varsigma tau upsilon phi varphi chi psi omega Gamma Delta Theta Lambda Xi Pi Sigma
Upsilon Phi Psi Omega digamma aleph beth gimel hbar ell wp Re Im partial infty nabla emptyset varnothing forall
exists nexists neg lnot top bot angle triangle backslash prime imath jmath Box Diamond surd flat natural sharp
clubsuit diamondsuit heartsuit spadesuit complement eth mho
pm mp times div cdot ast star circ bullet cap cup uplus sqcap sqcup vee wedge lor land setminus wr diamond
bigtriangleup bigtriangledown triangleleft triangleright oplus ominus otimes oslash odot bigcirc dagger ddagger
amalg ltimes rtimes smallsetminus
leq le geq ge neq ne equiv sim simeq approx cong asymp propto models perp mid parallel nmid nparallel subset
supset subseteq supseteq subsetneq supsetneq nsubseteq in notin ni owns prec succ preceq succeq ll gg vdash dashv
doteq bowtie smile frown lessgtr leqslant geqslant lesssim gtrsim approxeq ncong nsim not coloneqq
leftarrow rightarrow to gets leftrightarrow Leftarrow Rightarrow Leftrightarrow longleftarrow longrightarrow
longleftrightarrow Longleftarrow Longrightarrow Longleftrightarrow mapsto longmapsto hookleftarrow hookrightarrow
uparrow downarrow updownarrow Uparrow Downarrow Updownarrow nearrow searrow swarrow nwarrow iff implies impliedby
rightleftharpoons leftharpoonup rightharpoonup xrightarrow xleftarrow circlearrowleft circlearrowright
twoheadrightarrow rightrightarrows leadsto
sum prod coprod int iint iiint oint bigcup bigcap bigvee bigwedge bigoplus bigotimes bigodot biguplus bigsqcup
sin cos tan cot sec csc arcsin arccos arctan sinh cosh tanh coth log ln lg exp det dim ker hom arg deg gcd lcm
lim liminf limsup sup inf max min Pr mod bmod pmod pod operatorname
frac dfrac tfrac cfrac binom dbinom tbinom choose sqrt root over atop above
left right big Big bigg Bigg bigl bigr Bigl Bigr biggl biggr Biggl Biggr bigm Bigm biggm Biggm middle
because therefore fbox framebox
langle rangle lfloor rfloor lceil rceil lvert rvert lVert rVert vert Vert lbrace rbrace lbrack rbrack
hat widehat tilde widetilde bar overline underline vec dot ddot dddot acute grave breve check mathring
overbrace underbrace overset underset stackrel overleftarrow overrightarrow overleftrightarrow underrightarrow
boxed cancel bcancel xcancel
mathbf mathrm mathit mathsf mathtt mathcal mathbb mathfrak mathscr boldsymbol bm pmb mathnormal Bbb
text textbf textit textrm textsf texttt textnormal mbox hbox emph rm bf it sf tt cal
displaystyle textstyle scriptstyle scriptscriptstyle tiny scriptsize small normalsize large Large LARGE huge Huge
quad qquad enspace thinspace negthinspace hspace hskip kern mkern mspace space nobreakspace phantom hphantom
vphantom smash rlap llap mathrlap mathllap mathclap clap strut
color textcolor colorbox fcolorbox definecolor
begin end matrix pmatrix bmatrix Bmatrix vmatrix Vmatrix smallmatrix array aligned alignedat gathered split cases
dcases rcases align gather equation multline subarray substack
hline hdashline cline cr newline tag label ref eqref nonumber notag intertext
ldots cdots vdots ddots dots dotsc dotsb dotsm dotsi dotso iddots
newcommand renewcommand def let DeclareMathOperator
mathop mathbin mathrel mathopen mathclose mathpunct mathord mathinner limits nolimits
unicode href class style cssId mathchoice bbox enclose toggle raise lower raisebox moveleft moveright
lt gt amp colon vcenter halign noalign openup tabskip hfil hfill
""".split())


def unknown_commands(text: str, known: frozenset[str] = MATHJAX_COMMANDS) -> Counter:
    """Control words in `text` that MathJax may not know."""
    tokens = re.findall(r"\\([A-Za-z]+|.)", text)  # \\c is a row break and c, not \c
    return Counter(t for t in tokens if t[0].isalpha() and t not in known)
