"""Atomic result writes for the video-fix studies: write to a temp file next to the target, then os.replace.
A run killed mid-write leaves the previous complete file (or none), never a truncated one."""
from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path


def _tmp(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp{os.getpid()}")


def write_text(path, text: str) -> None:
    path = Path(path)
    tmp = _tmp(path)
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_json(path, obj, **kw) -> None:
    kw.setdefault("indent", 1)
    write_text(path, json.dumps(obj, **kw))


def write_csv(path, rows, fieldnames=None) -> None:
    """rows: list of dicts (fieldnames from the first row unless given) or, with fieldnames=None and
    rows[0] a list, a list of lists whose first row is the header."""
    buf = io.StringIO()
    if rows and isinstance(rows[0], dict):
        w = csv.DictWriter(buf, fieldnames=fieldnames or list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    else:
        w = csv.writer(buf)
        w.writerows(rows)
    write_text(path, buf.getvalue())


def savefig(fig, path, **kw) -> None:
    path = Path(path)
    tmp = path.with_name(f".{path.stem}.tmp{os.getpid()}{path.suffix}")
    fig.savefig(tmp, **kw)
    os.replace(tmp, path)
