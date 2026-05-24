"""Feature registry. Same lazy-import pattern as imagery and
ground_truth packages."""

from .base import FeatureSpec

__all__ = ["FeatureSpec", "get_feature"]


def get_feature(name: str, **kwargs) -> FeatureSpec:
    if name == "building":
        from .building import BuildingFeature
        return BuildingFeature(**kwargs)
    if name == "park":
        from .park import ParkFeature
        return ParkFeature(**kwargs)
    if name == "road":
        from .road import RoadFeature
        return RoadFeature(**kwargs)
    if name == "parking":
        from .parking import ParkingFeature
        return ParkingFeature(**kwargs)
    raise KeyError(
        f"unknown feature: {name!r}. Known: 'building', 'park', 'road', 'parking'."
    )
