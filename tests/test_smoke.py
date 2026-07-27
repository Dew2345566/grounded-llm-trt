"""Placeholder test so CI is green from commit one. Replaced with real tests in Phase 1."""


def test_imports():
    import src  # noqa: F401


def test_repo_layout():
    from pathlib import Path
    for d in ["configs", "src/training", "src/rag", "src/deploy", "src/eval"]:
        assert Path(d).is_dir(), f"missing directory: {d}"
