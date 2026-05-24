"""Recipe + base-class shape tests.

These don't hit the network. We construct fake imagery / GT / feature
implementations purely to exercise the Recipe wiring, the lazy
get_source() registries, and load_recipe() file loading.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from rasterio.transform import from_bounds
from shapely.geometry import Polygon

from imagery_seg.features.base import FeatureSpec
from imagery_seg.ground_truth.base import GroundTruthSource
from imagery_seg.imagery.base import FetchedImage, ImagerySource
from imagery_seg.recipe import AOI, Recipe, TrainingConfig, load_recipe


class _FakeImagery(ImagerySource):
    name = "fake_img"

    def fetch_for_bbox(self, bbox, max_side=1024, dst_crs="EPSG:3857"):
        arr = np.zeros((3, 4, 4), dtype=np.uint8)
        return FetchedImage(
            array=arr,
            transform=from_bounds(*bbox, width=4, height=4),
            crs=dst_crs,
            asset_id="fake://item",
        )


class _FakeGT(GroundTruthSource):
    name = "fake_gt"

    def fetch_polygons(self, bbox, feature_query):
        west, south, east, north = bbox
        return [Polygon([(west, south), (east, south), (east, north), (west, north)])]


@dataclass(frozen=True)
class _FakeFeature(FeatureSpec):
    def overpass_query(self, bbox):
        return f"way[fake];out;{bbox}"


def _recipe() -> Recipe:
    return Recipe(
        name="unit",
        imagery=_FakeImagery(),
        ground_truth=_FakeGT(),
        feature=_FakeFeature(name="fake_feat"),
        training=TrainingConfig(epochs=1, batch_size=1),
        aois=((0.0, 0.0, 1.0, 1.0),),
    )


def test_fetched_image_dims():
    img = _FakeImagery().fetch_for_bbox((0, 0, 1, 1))
    assert img.height == 4
    assert img.width == 4
    assert img.array.shape == (3, 4, 4)
    assert img.crs == "EPSG:3857"


def test_recipe_cache_namespace():
    r = _recipe()
    assert r.cache_namespace == "fake_img__fake_gt__fake_feat"


def test_split_cache_namespaces():
    r = _recipe()
    assert r.imagery_cache_namespace == "fake_img"
    assert r.polygon_cache_namespace == "fake_gt__fake_feat"


def test_recipe_aois_tuple():
    r = _recipe()
    # Bare bbox tuples passed in get coerced to AOI(..., region="")
    assert len(r.aois) == 1
    assert isinstance(r.aois[0], AOI)
    assert r.aois[0].bbox == (0.0, 0.0, 1.0, 1.0)
    assert r.aois[0].region == ""


def test_feature_vector_tile_default_raises():
    f = _FakeFeature(name="fake_feat")
    with pytest.raises(NotImplementedError):
        f.vector_tile_layer()


def test_load_recipe_from_file(tmp_path: Path):
    src = tmp_path / "tiny.py"
    src.write_text(
        "from imagery_seg.recipe import Recipe, TrainingConfig\n"
        "from tests.test_recipe import _FakeImagery, _FakeGT, _FakeFeature\n"
        "recipe = Recipe(\n"
        "    name='loaded',\n"
        "    imagery=_FakeImagery(),\n"
        "    ground_truth=_FakeGT(),\n"
        "    feature=_FakeFeature(name='loaded_feat'),\n"
        "    aois=((0.0, 0.0, 1.0, 1.0),),\n"
        ")\n",
        encoding="utf-8",
    )
    r = load_recipe(src)
    assert r.name == "loaded"
    assert r.feature.name == "loaded_feat"


def test_load_recipe_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_recipe(tmp_path / "nope.py")


def test_load_recipe_missing_attr(tmp_path: Path):
    src = tmp_path / "empty.py"
    src.write_text("# no recipe defined\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_recipe(src)


def test_effective_split_by_val_regions():
    """val_regions takes priority: every AOI whose region is listed is val."""
    a1 = AOI(bbox=(0.0, 0.0, 1.0, 1.0), region="jp")
    a2 = AOI(bbox=(1.0, 1.0, 2.0, 2.0), region="jp")
    a3 = AOI(bbox=(5.0, 5.0, 6.0, 6.0), region="ph")
    r = Recipe(
        name="explicit",
        imagery=_FakeImagery(),
        ground_truth=_FakeGT(),
        feature=_FakeFeature(name="fake_feat"),
        aois=(a1, a2, a3),
        val_regions=("ph",),
    )
    assert r.effective_train_aois == (a1, a2)
    assert r.effective_val_aois == (a3,)


def test_effective_split_implicit_last_as_val():
    a1 = AOI(bbox=(0.0, 0.0, 1.0, 1.0), region="")
    a2 = AOI(bbox=(1.0, 1.0, 2.0, 2.0), region="")
    r = Recipe(
        name="implicit",
        imagery=_FakeImagery(),
        ground_truth=_FakeGT(),
        feature=_FakeFeature(name="fake_feat"),
        aois=(a1, a2),
    )
    assert r.effective_train_aois == (a1,)
    assert r.effective_val_aois == (a2,)


def test_from_spec_coerces_bare_bbox():
    """from_spec accepts (bbox tuple) for one-off recipes; region is ''."""
    from imagery_seg.imagery.base import ImagerySource
    # We don't go through the registry here because we'd need real names;
    # exercise the Recipe constructor coercion directly via __post_init__.
    r = Recipe(
        name="loose",
        imagery=_FakeImagery(),
        ground_truth=_FakeGT(),
        feature=_FakeFeature(name="f"),
        aois=((0.0, 0.0, 1.0, 1.0), AOI(bbox=(2.0, 2.0, 3.0, 3.0), region="jp")),
    )
    assert len(r.aois) == 2
    assert all(isinstance(a, AOI) for a in r.aois)
    assert r.aois[0].region == ""  # coerced
    assert r.aois[1].region == "jp"  # passed through


def test_holdout_flag_marks_aoi_as_val():
    """Within-region split via AOI.holdout=True (no val_regions needed)."""
    a1 = AOI(bbox=(0.0, 0.0, 1.0, 1.0), region="jp")
    a2 = AOI(bbox=(1.0, 1.0, 2.0, 2.0), region="jp")
    a3 = AOI(bbox=(2.0, 2.0, 3.0, 3.0), region="jp", holdout=True)
    r = Recipe(
        name="hold",
        imagery=_FakeImagery(),
        ground_truth=_FakeGT(),
        feature=_FakeFeature(name="f"),
        aois=(a1, a2, a3),
    )
    assert r.effective_train_aois == (a1, a2)
    assert r.effective_val_aois == (a3,)


def test_holdout_and_val_regions_compose_via_or():
    a1 = AOI(bbox=(0, 0, 1, 1), region="jp")
    a2 = AOI(bbox=(1, 1, 2, 2), region="jp", holdout=True)
    a3 = AOI(bbox=(2, 2, 3, 3), region="ph")
    r = Recipe(
        name="mix",
        imagery=_FakeImagery(),
        ground_truth=_FakeGT(),
        feature=_FakeFeature(name="f"),
        aois=(a1, a2, a3),
        val_regions=("ph",),
    )
    # a1 = train. a2 = val (holdout). a3 = val (val_regions).
    assert r.effective_train_aois == (a1,)
    assert set(r.effective_val_aois) == {a2, a3}


def test_regions_property_is_sorted_unique_nonempty():
    a1 = AOI(bbox=(0, 0, 1, 1), region="jp")
    a2 = AOI(bbox=(1, 1, 2, 2), region="jp")
    a3 = AOI(bbox=(5, 5, 6, 6), region="ph")
    a4 = AOI(bbox=(7, 7, 8, 8), region="")
    r = Recipe(
        name="r",
        imagery=_FakeImagery(),
        ground_truth=_FakeGT(),
        feature=_FakeFeature(name="f"),
        aois=(a1, a2, a3, a4),
    )
    assert r.regions == ("jp", "ph")


def test_effective_split_single_aoi_no_val():
    r = _recipe()  # 1 AOI total
    assert r.effective_train_aois == r.aois
    assert r.effective_val_aois == ()


def test_run_dir_combines_output_dir_and_name(tmp_path: Path):
    from imagery_seg.recipe import TrainingConfig as TC
    r = Recipe(
        name="abc",
        imagery=_FakeImagery(),
        ground_truth=_FakeGT(),
        feature=_FakeFeature(name="f"),
        training=TC(output_dir=str(tmp_path / "runs")),
        aois=((0.0, 0.0, 1.0, 1.0),),
    )
    assert r.run_dir == tmp_path / "runs" / "abc"


def test_get_source_unknown():
    from imagery_seg.imagery import get_source as get_img
    from imagery_seg.ground_truth import get_source as get_gt
    from imagery_seg.features import get_feature

    with pytest.raises(KeyError):
        get_img("not_a_real_source")
    with pytest.raises(KeyError):
        get_gt("not_a_real_source")
    with pytest.raises(KeyError):
        get_feature("not_a_real_feature")
