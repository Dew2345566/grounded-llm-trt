"""
Phase 1, Stage 2 — Parse ROS2 reStructuredText into clean, section-level documents.

Design notes
------------
Sphinx .rst is not plain text: it carries directives (`.. toctree::`), roles
(``:doc:`link```), and inline markup that add tokens without adding meaning.
Embedding that noise makes retrieval worse, so it is stripped here.

Two things are deliberately NOT stripped:

1. Code blocks. For a ROS2 assistant, the shell command *is* the answer. They are
   unindented and fenced so the LLM sees them as code, not prose.
2. Section headers. RST's underline convention encodes a document hierarchy
   (= then - then ^ then ~), which gives natural chunk boundaries and a
   breadcrumb trail that goes into chunk metadata. A passage that says
   "run `colcon build`" is ambiguous; one titled
   "Tutorials > Beginner > Using colcon > Build the workspace" is not.

Run:
    python data/parse_rst.py --src data/raw/ros2_docs/source --out data/processed/sections.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

# RST section underlines, in the order Sphinx conventionally nests them.
# The actual level is assigned by order of first appearance per file, since
# authors are not perfectly consistent.
UNDERLINE_CHARS = '=-~^"\'`#*+_:.'

# Directives whose *content* we want to drop entirely (noise or navigation).
DROP_DIRECTIVES = {
    "toctree", "contents", "redirect-from", "index", "only",
    "raw", "meta", "sectionauthor", "highlight",
}

# Directives whose content we keep, treated as code.
CODE_DIRECTIVES = {"code-block", "code", "literalinclude", "parsed-literal"}

# Directives whose content we keep as prose, with a label prefix.
ADMONITION_DIRECTIVES = {
    "note", "warning", "tip", "important", "caution", "attention", "seealso",
}


@dataclass
class Section:
    """One document section: a header plus the prose and code beneath it."""
    doc_id: str          # relative path, e.g. "Tutorials/Colcon-Tutorial.rst"
    title: str           # this section's header
    breadcrumb: str      # "Using colcon > Prerequisites > Install colcon"
    level: int           # 0 = page title
    text: str            # cleaned body
    n_chars: int
    has_code: bool
    url: str             # best-guess docs.ros.org URL for citation


# ---------------------------------------------------------------------------
# Inline cleanup
# ---------------------------------------------------------------------------

def clean_inline(text: str) -> str:
    """Strip RST inline markup while preserving the words themselves."""
    # `label <url>`__  ->  label (url)
    text = re.sub(r"`([^`<]+?)\s*<([^>]+)>`__?", r"\1 (\2)", text)
    # :doc:`label <path>` / :ref:`label <target>`  ->  label
    text = re.sub(r":\w+:`([^`<]+?)\s*<[^>]+>`", r"\1", text)
    # :doc:`target`  ->  target
    text = re.sub(r":\w+:`([^`]+)`", r"\1", text)
    # ``literal``  ->  `literal`   (markdown-style, fewer tokens)
    text = re.sub(r"``([^`]+)``", r"`\1`", text)
    # **bold** and *italic* -> plain
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", text)
    # Anchors: .. _Label:
    text = re.sub(r"^\.\.\s+_[^:]+:\s*$", "", text, flags=re.M)
    # Substitutions |name|
    text = re.sub(r"\|([^|]+)\|", r"\1", text)
    return text


def clean_inline_outside_code(lines: list[str]) -> list[str]:
    """
    Apply inline cleanup line-by-line, leaving fenced code untouched.

    Ordering matters: cleaning the whole document at once lets the ``literal``
    pattern match across a ``` fence and destroy it. Shell globs (*) and paths
    (**) inside code would also be misread as RST emphasis.
    """
    out, in_code = [], False
    for ln in lines:
        if ln.startswith("```"):
            in_code = not in_code
            out.append(ln)
            continue
        out.append(ln if in_code else clean_inline(ln))
    return out


def dedent_block(lines: list[str]) -> list[str]:
    """Remove the common leading indentation from a directive body."""
    non_empty = [ln for ln in lines if ln.strip()]
    if not non_empty:
        return []
    pad = min(len(ln) - len(ln.lstrip()) for ln in non_empty)
    return [ln[pad:] if len(ln) >= pad else ln for ln in lines]


# ---------------------------------------------------------------------------
# Directive handling
# ---------------------------------------------------------------------------

DIRECTIVE_RE = re.compile(r"^(\s*)\.\.\s+([\w-]+)::\s*(.*)$")


def process_directives(lines: list[str]) -> list[str]:
    """
    Walk the file and resolve directives: drop navigation, fence code,
    label admonitions, flatten tabs into labelled prose.
    """
    out: list[str] = []
    i = 0
    while i < len(lines):
        m = DIRECTIVE_RE.match(lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue

        indent, name, arg = m.group(1), m.group(2).lower(), m.group(3).strip()
        base_indent = len(indent)

        # Collect the directive body: everything more-indented than the directive,
        # plus blank lines.
        j = i + 1
        body: list[str] = []
        while j < len(lines):
            ln = lines[j]
            if not ln.strip():
                body.append("")
                j += 1
                continue
            if len(ln) - len(ln.lstrip()) > base_indent:
                body.append(ln)
                j += 1
            else:
                break

        # Drop directive options (:depth: 2, :language: python, ...)
        body = [b for b in body if not re.match(r"^\s*:[\w-]+:", b)]
        body = dedent_block(body)

        if name in DROP_DIRECTIVES:
            pass  # emit nothing

        elif name in CODE_DIRECTIVES:
            lang = arg if arg and arg not in {"console", "bash", "sh"} else "bash"
            # Trim blank padding so the fence hugs the actual code.
            trimmed = body[:]
            while trimmed and not trimmed[0].strip():
                trimmed.pop(0)
            while trimmed and not trimmed[-1].strip():
                trimmed.pop()
            if trimmed:
                out.append(f"```{lang}")
                out.extend(trimmed)
                out.append("```")

        elif name in ADMONITION_DIRECTIVES:
            out.append(f"{name.upper()}: {arg}".rstrip(": "))
            out.extend(body)

        elif name == "group-tab" or name == "tab":
            # Per-platform variants: keep, but label so the model knows the OS.
            out.append(f"[{arg}]")
            out.extend(process_directives(body))

        elif name == "tabs":
            out.extend(process_directives(body))

        elif name == "image" or name == "figure":
            pass  # no visual channel in this pipeline

        else:
            # Unknown directive: keep the body, drop the marker.
            out.extend(process_directives(body))

        i = j
    return out


# ---------------------------------------------------------------------------
# Section splitting
# ---------------------------------------------------------------------------

def is_underline(line: str, prev: str) -> bool:
    """True if `line` is an RST section underline for `prev`."""
    s = line.rstrip()
    if len(s) < 3 or not prev.strip():
        return False
    if s[0] not in UNDERLINE_CHARS:
        return False
    if len(set(s)) != 1:
        return False
    # Underline must be at least as long as the title (allow small slack).
    return len(s) >= len(prev.rstrip()) - 2


def split_sections(lines: list[str]) -> list[tuple[str, int, list[str]]]:
    """
    Split into (title, level, body_lines).
    Level is assigned by order of first appearance of each underline char.
    """
    char_levels: dict[str, int] = {}
    sections: list[tuple[str, int, list[str]]] = []
    current_title, current_level, buf = "", 0, []

    i = 0
    in_code = False
    while i < len(lines):
        if lines[i].startswith("```"):
            in_code = not in_code
            buf.append(lines[i])
            i += 1
            continue
        # A row of dashes inside a code block is not a section underline.
        if not in_code and i + 1 < len(lines) and is_underline(lines[i + 1], lines[i]):
            # flush previous
            if current_title or buf:
                sections.append((current_title, current_level, buf))
            title = lines[i].strip()
            ch = lines[i + 1].strip()[0]
            if ch not in char_levels:
                char_levels[ch] = len(char_levels)
            current_title, current_level, buf = title, char_levels[ch], []
            i += 2
            continue
        buf.append(lines[i])
        i += 1

    if current_title or buf:
        sections.append((current_title, current_level, buf))
    return sections


def build_breadcrumb(stack: list[str]) -> str:
    return " > ".join(s for s in stack if s)


def rst_to_url(rel_path: Path) -> str:
    """Best-effort docs.ros.org URL so retrieved chunks can cite a source."""
    stem = str(rel_path.with_suffix("")).replace("\\", "/")
    return f"https://docs.ros.org/en/rolling/{stem}.html"


# ---------------------------------------------------------------------------
# Per-file pipeline
# ---------------------------------------------------------------------------

def parse_file(path: Path, src_root: Path) -> list[Section]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    lines = raw.splitlines()

    lines = process_directives(lines)
    lines = clean_inline_outside_code(lines)

    rel = path.relative_to(src_root)
    url = rst_to_url(rel)

    out: list[Section] = []
    stack: list[str] = []

    for title, level, body in split_sections(lines):
        # Maintain the header stack for breadcrumbs.
        if title:
            stack = stack[:level]
            stack.append(title)

        body_text = "\n".join(body)
        # Collapse 3+ blank lines, trim trailing whitespace per line.
        body_text = re.sub(r"\n{3,}", "\n\n", body_text)
        body_text = "\n".join(ln.rstrip() for ln in body_text.splitlines()).strip()

        if len(body_text) < 40:  # headers with no real content
            continue

        out.append(Section(
            doc_id=str(rel),
            title=title or rel.stem,
            breadcrumb=build_breadcrumb(stack) or rel.stem,
            level=level,
            text=body_text,
            n_chars=len(body_text),
            has_code="```" in body_text,
            url=url,
        ))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/raw/ros2_docs/source")
    ap.add_argument("--out", default="data/processed/sections.jsonl")
    ap.add_argument("--preview", type=int, default=0,
                    help="Print N sample sections instead of writing output")
    args = ap.parse_args()

    src_root = Path(args.src)
    if not src_root.exists():
        raise SystemExit(f"Source not found: {src_root}. Clone the docs first.")

    files = sorted(src_root.rglob("*.rst"))
    all_sections: list[Section] = []
    for f in files:
        try:
            all_sections.extend(parse_file(f, src_root))
        except Exception as e:  # keep going; report at the end
            print(f"  ! failed {f}: {e}")

    if args.preview:
        for s in all_sections[:args.preview]:
            print("=" * 70)
            print(f"{s.breadcrumb}   [level {s.level}, {s.n_chars} chars, code={s.has_code}]")
            print("-" * 70)
            print(s.text[:800])
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for s in all_sections:
            fh.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")

    total_chars = sum(s.n_chars for s in all_sections)
    with_code = sum(1 for s in all_sections if s.has_code)
    print(f"Files parsed:      {len(files)}")
    print(f"Sections written:  {len(all_sections)}")
    print(f"With code blocks:  {with_code} ({with_code / max(len(all_sections),1):.0%})")
    print(f"Total characters:  {total_chars:,}  (~{total_chars // 4:,} tokens)")
    print(f"Mean section size: {total_chars // max(len(all_sections),1):,} chars")
    print(f"Output:            {out_path}")


if __name__ == "__main__":
    main()