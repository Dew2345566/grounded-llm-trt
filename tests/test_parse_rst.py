"""Unit tests for the RST parser. These pin down the two bugs found in session 2."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.parse_rst import (  # noqa: E402
    clean_inline,
    clean_inline_outside_code,
    process_directives,
    split_sections,
    is_underline,
)


def test_inline_link_becomes_text_and_url():
    out = clean_inline("see `this doc <https://example.com>`__ now")
    assert "this doc" in out and "https://example.com" in out
    assert "`" not in out


def test_literal_becomes_single_backtick():
    assert clean_inline("run ``colcon build``") == "run `colcon build`"


def test_code_fence_survives_inline_cleaning():
    """Regression: the ``literal`` pattern used to eat ``` fences."""
    lines = ["```bash", "$ ls **/*.py", "```"]
    out = clean_inline_outside_code(lines)
    assert out[0] == "```bash"
    assert out[1] == "$ ls **/*.py"  # glob must not be treated as emphasis
    assert out[2] == "```"


def test_code_block_directive_is_fenced():
    lines = [".. code-block:: console", "", "    $ ros2 run demo talker", ""]
    out = process_directives(lines)
    assert out[0].startswith("```")
    assert "$ ros2 run demo talker" in out[1]
    assert out[-1] == "```"


def test_toctree_is_dropped():
    lines = [".. toctree::", "   :maxdepth: 2", "", "   Some-Page", "", "Real text"]
    out = process_directives(lines)
    assert "Some-Page" not in "\n".join(out)
    assert "Real text" in "\n".join(out)


def test_underline_detection():
    assert is_underline("=====", "Title")
    assert not is_underline("=-=-=", "Title")   # mixed chars
    assert not is_underline("==", "A Long Title")  # too short


def test_dashes_inside_code_do_not_split_sections():
    """Regression: a row of dashes in a code block looked like a header."""
    lines = ["Title", "=====", "text", "```bash", "output", "------", "```", "more"]
    secs = split_sections(lines)
    assert len(secs) == 1
    assert secs[0][0] == "Title"