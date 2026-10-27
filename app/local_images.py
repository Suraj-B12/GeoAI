"""
Local image store paths (DB_BACKEND=postgres): the files that replace Cloudinary.

With the local backend an image is a PNG on this machine, and the database
column that held the Cloudinary URL (photos.image_url, assessments.image_url)
holds its ABSOLUTE path instead, e.g. C:\\GeoAI\\images\\<session>\\<segment>.png.
Code that reads image_url tells the two apart with `local_image_path()`:

    https://res.cloudinary.com/...      -> None (a URL: download it as before)
    C:\\GeoAI\\images\\s1\\s1_seg00003.png -> Path (read / delete the file)

Every read and delete is confined to LOCAL_IMAGE_ROOT. image_url is written by
outside code too (the RoadSide app writes photos.image_url on Supabase), so a
row must never be able to make the server read or delete an arbitrary file:
paths outside the root raise PermissionError, whatever the backend.

Standard library only: app/supabase_client.py and app/deletion.py import this
in production, where the Supabase backend must not depend on psycopg.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

DEFAULT_IMAGE_ROOT = r"C:\GeoAI\images"
_URL = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")


def image_root() -> Path:
    """LOCAL_IMAGE_ROOT (default C:\\GeoAI\\images), read at call time."""
    return Path(os.environ.get("LOCAL_IMAGE_ROOT") or DEFAULT_IMAGE_ROOT)


def local_image_path(url: Optional[str]) -> Optional[Path]:
    """The absolute local path an image_url names, or None if it is a URL / empty / relative.

    A Windows path such as C:\\x parses as a URL with scheme "c" under urlparse,
    so a URL is recognised by "<scheme>://" instead.
    """
    if not url or not isinstance(url, str) or _URL.match(url.strip()):
        return None
    p = Path(url.strip())
    return p if p.is_absolute() else None


def inside_root(path: Path, root: Optional[Path] = None) -> Path:
    """`path` resolved, if it lies under the image root; PermissionError otherwise.

    resolve() removes "..", so C:\\GeoAI\\images\\..\\secret is caught.
    WindowsPath comparisons are case-insensitive, as the file system is.
    """
    root_r = (root or image_root()).resolve()
    p = Path(path).resolve()
    if p != root_r and root_r in p.parents:
        return p
    raise PermissionError(f"image path {str(path)!r} is outside LOCAL_IMAGE_ROOT ({root_r})")


def read_local_image(url_or_path, root: Optional[Path] = None) -> bytes:
    """Bytes of a stored image. FileNotFoundError if it is gone, PermissionError outside the root."""
    p = local_image_path(str(url_or_path))
    if p is None:
        raise ValueError(f"not a local image path: {url_or_path!r}")
    return inside_root(p, root).read_bytes()


def delete_local_image(url_or_path, root: Optional[Path] = None) -> dict:
    """Delete a stored image. {"path", "result": "deleted" | "already_gone"}.

    A missing file is "already_gone": unlike a Cloudinary "not found" there is no
    CDN copy that could still be served, so it is safe to treat as done.
    Raises PermissionError outside the root and OSError when the file stays.
    """
    p = local_image_path(str(url_or_path))
    if p is None:
        raise ValueError(f"not a local image path: {url_or_path!r}")
    p = inside_root(p, root)
    try:
        p.unlink()
        result = "deleted"
    except FileNotFoundError:
        result = "already_gone"
    if p.exists():
        raise OSError(f"{p} still exists after delete")
    return {"path": str(p), "result": result}
