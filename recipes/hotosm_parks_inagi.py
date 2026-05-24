"""Recipe: HOTOSM Inagi parks — compact high-resolution suburban park tile.

Backed by HOTOSM item `617e9d7d8ebcb9000515c520` (`inagi202109`),
captured on 2021-09-27 with ~4.45 cm GSD by DRONEBIRD/Nashitakahashi.

This tile is attractive because it is small, visually clean, and rich in
named park polygons while still being peacetime-style imagery rather than
obvious post-disaster capture. We therefore use it as a second HOTOSM park
recipe after Sagamihara.

AOIs are aligned to a simple 3×3 scout grid over the item bbox. We choose
only cells with 2-3 park polygons so the recipe stays comparable across
train/val and remains traceable back to `aoi-grid` output.
"""

from imagery_seg.recipe import AOI, Recipe, TrainingConfig

_W, _S, _E, _N = 139.489636, 35.631850, 139.504601, 35.644067
_DLON = (_E - _W) / 3
_DLAT = (_N - _S) / 3


def _cell(i: int, j: int) -> tuple[float, float, float, float]:
    return (_W + i * _DLON, _S + j * _DLAT, _W + (i + 1) * _DLON, _S + (j + 1) * _DLAT)


recipe = Recipe.from_spec(
    name="hotosm_parks_inagi",
    imagery="hotosm",
    ground_truth="overpass",
    feature="park",
    feature_kwargs={
        "min_positive_polygon_count": 2,
    },
    training=TrainingConfig(
        epochs=50,
        batch_size=2,
        lr=1e-4,
        encoder="resnet34",
        encoder_weights="imagenet",
        max_side=1024,
    ),
    aois=(
        AOI(
            bbox=_cell(0, 0),
            region="japan",
            phenotype="japan_suburban_park",
            notes="c0r0: south-west cell; includes 稲城中央公園 cluster, 2 park polygons",
            source_item="617e9d7d8ebcb9000515c520",
            imagery_year=2021,
        ),
        AOI(
            bbox=_cell(2, 0),
            region="japan",
            phenotype="japan_suburban_park",
            notes="c2r0: south-east cell; 3 park polygons around eastern residential edge",
            source_item="617e9d7d8ebcb9000515c520",
            imagery_year=2021,
        ),
        AOI(
            bbox=_cell(0, 2),
            region="japan",
            phenotype="japan_suburban_park",
            notes="c0r2: north-west cell; 城山公園 side, 3 park polygons",
            source_item="617e9d7d8ebcb9000515c520",
            imagery_year=2021,
        ),
        AOI(
            bbox=_cell(1, 1),
            region="japan",
            phenotype="japan_suburban_park",
            notes="c1r1: central holdout cell; 1-2 larger neighbourhood parks, val holdout",
            source_item="617e9d7d8ebcb9000515c520",
            imagery_year=2021,
            holdout=True,
        ),
    ),
)
