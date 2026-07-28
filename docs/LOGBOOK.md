# Development Logbook

One entry per working session. Useful for the README write-up later, and for remembering
why something was done a certain way three weeks after doing it.

## Session template
**Date / duration:**
**Goal:**
**Done:**
**Blocked on:**
**Next session starts with:**

---

## 2026-07-27 — Session 1
**Goal:** Repository scaffold and project setup.
**Done:** Directory structure, README skeleton, .gitignore, pinned requirements, docs stubs.
**Blocked on:** —
**Next session starts with:** Cloning ROS2 docs source and inspecting .rst structure.

## 2026-07-28 — Session 2
Done: cloned ROS2 docs (334 .rst), wrote data/parse_rst.py.
Corpus: 4,707 sections, ~993k tokens, 24% contain code.
Sizes: median 507 chars, p95 2,480, max 29,769.
Friction: most of the session went to Windows path/download issues,
not the code. Fix: get git push working.
Next: chunking — split 363 oversized sections, merge tiny ones.