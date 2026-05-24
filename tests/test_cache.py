"""Cache primitives: key composition, atomic write, file lock."""

from __future__ import annotations

import threading
from pathlib import Path

from imagery_seg.cache import cache_key, file_lock, write_atomic


def test_cache_key_is_deterministic_and_short():
    k1 = cache_key("hotosm", "ITEM1", bbox=(0.0, 0.0, 1.0, 1.0), max_side=256)
    k2 = cache_key("hotosm", "ITEM1", bbox=(0.0, 0.0, 1.0, 1.0), max_side=256)
    assert k1 == k2
    # hex-digest based; len is bounded
    assert 12 <= len(k1) <= 64
    assert k1.isalnum()


def test_cache_key_differs_on_any_axis():
    base = cache_key("hotosm", "ITEM1", bbox=(0.0, 0.0, 1.0, 1.0), max_side=256)
    assert base != cache_key("hotosm", "ITEM2", bbox=(0.0, 0.0, 1.0, 1.0), max_side=256)
    assert base != cache_key("hotosm", "ITEM1", bbox=(0.0, 0.0, 1.0, 1.1), max_side=256)
    assert base != cache_key("hotosm", "ITEM1", bbox=(0.0, 0.0, 1.0, 1.0), max_side=512)
    assert base != cache_key("sentinel2", "ITEM1", bbox=(0.0, 0.0, 1.0, 1.0), max_side=256)


def test_write_atomic_creates_file(tmp_path: Path):
    target = tmp_path / "out.bin"
    write_atomic(target, b"hello")
    assert target.read_bytes() == b"hello"


def test_write_atomic_no_partial_file_visible(tmp_path: Path):
    """Tmp file shouldn't linger after a successful write."""
    target = tmp_path / "out.bin"
    write_atomic(target, b"hello")
    siblings = list(tmp_path.iterdir())
    assert siblings == [target]


def test_file_lock_serialises_writers(tmp_path: Path):
    target = tmp_path / "shared.txt"
    target.write_text("")
    lock_path = tmp_path / "shared.lock"

    log: list[str] = []

    def worker(tag: str) -> None:
        with file_lock(lock_path):
            log.append(f"enter:{tag}")
            # Hold long enough to overlap if locking is broken
            import time as _t
            _t.sleep(0.05)
            log.append(f"exit:{tag}")

    threads = [threading.Thread(target=worker, args=(t,)) for t in "AB"]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Critical sections must not interleave: enter must immediately
    # precede exit of the *same* tag.
    for i in range(0, len(log), 2):
        a, b = log[i], log[i + 1]
        assert a.startswith("enter:") and b.startswith("exit:")
        assert a.split(":")[1] == b.split(":")[1]
