"""Ground-truth source registry. See imagery/__init__.py for the
identical pattern."""

from .base import GroundTruthSource

__all__ = ["GroundTruthSource", "get_source"]


def get_source(name: str, **kwargs) -> GroundTruthSource:
    if name == "overpass":
        from .overpass import OverpassGT
        return OverpassGT(**kwargs)
    if name == "vector_tile":
        from .vector_tile import VectorTileGT
        return VectorTileGT(**kwargs)
    raise KeyError(
        f"unknown ground-truth source: {name!r}. "
        "Known: 'overpass', 'vector_tile'."
    )
