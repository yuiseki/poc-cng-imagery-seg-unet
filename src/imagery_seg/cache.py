"""Filesystem-cache primitives shared by every backend.

- cache_key(...) -> stable hex digest from arbitrary kwargs.
- write_atomic(path, bytes) -> tmp file + os.replace so partial
  writes never become visible to readers.
- file_lock(path) -> fcntl-based exclusive lock that survives
  cross-process concurrency (i.e. two pytest workers or two server
  processes filling the same cache key at once).
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path


def cache_key(*positional: object, **named: object) -> str:
    """Compose a deterministic 32-char hex cache key from arbitrary args.

    Positional args go in first (most "primary" ordering), then named
    args in sorted order. Numeric floats are repr'd to preserve full
    precision so two near-identical bboxes don't collide.
    """
    parts = [_norm(p) for p in positional]
    for k in sorted(named):
        parts.append(f"{k}={_norm(named[k])}")
    payload = "|".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:32]


def _norm(value: object) -> str:
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_norm(v) for v in value) + "]"
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, default=_norm)
    return str(value)


def write_atomic(path: str | Path, data: bytes) -> None:
    """Write `data` to `path` atomically via a sibling tmp file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            with contextlib.suppress(FileNotFoundError):
                tmp.unlink()


@contextlib.contextmanager
def file_lock(lock_path: str | Path):
    """Exclusive lock held until the context exits.

    Uses fcntl.flock so we get cross-process exclusion (not just
    threading.Lock semantics). Lock file persists; flock state is
    released on fd close.
    """
    p = Path(lock_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(p, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


__all__ = ["cache_key", "file_lock", "write_atomic"]
