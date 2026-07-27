"""
Phase 1 — Build the ROS2 corpus and Q&A dataset.

Stage 1 (this session): clone/locate ROS2 documentation sources and inventory them.
Stage 2: chunk into passages with metadata.
Stage 3: generate synthetic Q&A pairs.

Run:
    python data/build_dataset.py --inventory
"""

import argparse
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def inventory(raw_dir: Path) -> None:
    """Report what source files are present, so chunking can be planned."""
    if not raw_dir.exists():
        print(f"{raw_dir} does not exist yet.")
        return

    rst_files = list(raw_dir.rglob("*.rst"))
    md_files = list(raw_dir.rglob("*.md"))
    print(f"Found {len(rst_files)} .rst and {len(md_files)} .md files under {raw_dir}")

    total_chars = sum(f.stat().st_size for f in rst_files + md_files)
    print(f"Approx corpus size: {total_chars / 1e6:.2f} MB")

    # TODO(session 2): print the section-header distribution to plan chunk boundaries.


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", action="store_true",
                        help="Report what is currently in data/raw/")
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if args.inventory:
        inventory(RAW_DIR)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
