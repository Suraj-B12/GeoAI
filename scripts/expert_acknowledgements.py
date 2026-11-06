"""
The experts to thank in the paper, from the review log (research.expert_review_log, local PostgreSQL).

Every expert who saved at least --min reviews is listed, alphabetically (an acknowledgement is not a
ranking), with the name exactly as they entered it on the expert page (the page asks for their real
full name and their consent to be named in the acknowledgements). Names are compared ignoring case
and repeated spaces, the same way as the page's leaderboard. Names listed in
paper/acknowledgement_exclude.txt (one per line: test accounts, people who withdrew) are left out.

    venv/Scripts/python.exe scripts/expert_acknowledgements.py            # print the list
    venv/Scripts/python.exe scripts/expert_acknowledgements.py --write    # update the paper's
        Acknowledgements section between <!-- EXPERT-ACK:START --> and <!-- EXPERT-ACK:END -->

Read-only on the database. Reads DATABASE_URL from the environment or .env; never prints it.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
PAPER = PROJECT_ROOT / "paper" / "baseline_evaluation.md"
EXCLUDE = PROJECT_ROOT / "paper" / "acknowledgement_exclude.txt"
START, END = "<!-- EXPERT-ACK:START -->", "<!-- EXPERT-ACK:END -->"


def norm(name: str) -> str:
    return " ".join(str(name or "").split()).lower()


def experts(conn, min_reviews: int) -> list[dict]:
    from app import expert_review as ER
    board = ER.leaderboard(conn)["experts"]
    skip = set()
    if EXCLUDE.exists():
        skip = {norm(x) for x in EXCLUDE.read_text(encoding="utf-8").splitlines() if x.strip() and not x.startswith("#")}
    out = [e for e in board if e["total"] >= min_reviews and norm(e["name"]) not in skip]
    return sorted(out, key=lambda e: norm(e["name"]))


def block(rows: list[dict]) -> str:
    if not rows:
        return (f"{START}\n*(No expert reviews are logged yet. Run `scripts/expert_acknowledgements.py --write` "
                f"once experts have reviewed.)*\n{END}")
    names = ", ".join(e["name"] for e in rows[:-1]) + (" and " if len(rows) > 1 else "") + rows[-1]["name"]
    total = sum(e["total"] for e in rows)
    return (f"{START}\nWe thank the pavement experts who reviewed the model's answers on the expert page: "
            f"{names}. Together they reviewed {total} road images (citizen photographs and drive-video frames); "
            f"their corrections are the ground truth of §6.4.\n{END}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--min", type=int, default=1, help="reviews an expert needs to be listed (default 1)")
    ap.add_argument("--write", action="store_true", help="update the paper's Acknowledgements block")
    a = ap.parse_args()
    if not os.environ.get("DATABASE_URL"):
        try:
            from dotenv import dotenv_values
            v = dotenv_values(PROJECT_ROOT / ".env").get("DATABASE_URL")
            if v:
                os.environ["DATABASE_URL"] = v
        except ImportError:
            pass
    from app import video_store
    with video_store.connect(connect_timeout=5) as conn:
        rows = experts(conn, a.min)
    for e in rows:
        print(f"{e['name']}\t{e['total']} reviewed ({e['citizen']} photos, {e['video']} video frames)")
    if not rows:
        print("no experts with reviews yet")
    text = block(rows)
    if a.write:
        paper = PAPER.read_text(encoding="utf-8")
        if START not in paper or END not in paper:
            print(f"markers {START} / {END} not found in {PAPER.name}", file=sys.stderr)
            return 1
        head, rest = paper.split(START, 1)
        _old, tail = rest.split(END, 1)
        PAPER.write_text(head + text + tail, encoding="utf-8")
        print(f"updated the Acknowledgements in {PAPER.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
