"""Imagery source registry.

Each concrete source (HOTOSM, Sentinel-2, ...) implements
ImagerySource and registers itself by a short name in the SOURCES dict
so recipes can refer to it textually without importing the class.
"""

from .base import FetchedImage, ImagerySource

__all__ = ["FetchedImage", "ImagerySource", "get_source"]


def get_source(name: str, **kwargs) -> ImagerySource:
    """Resolve a source name to an instance. Lazy imports so importing
    the registry doesn't pull in optional deps for every backend.
    """
    if name == "hotosm":
        from .hotosm import HotosmImagery
        return HotosmImagery(**kwargs)
    if name == "sentinel2":
        from .sentinel2 import Sentinel2Imagery
        return Sentinel2Imagery(**kwargs)
    raise KeyError(
        f"unknown imagery source: {name!r}. "
        "Known: 'hotosm', 'sentinel2'."
    )
