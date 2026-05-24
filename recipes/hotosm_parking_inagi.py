"""Recipe: HOTOSM Inagi parking — compact suburban parking cluster tile.

Also backed by HOTOSM item `617e9d7d8ebcb9000515c520` (`inagi202109`).

Why this recipe is useful:
  - it complements `hotosm_parking_isehara.py`, which is institutional /
    civic parking heavy;
  - Inagi's tile contains several small-to-medium suburban surface parking
    polygons, including restaurant/shop lots and park-side parking;
  - the tile is small enough that AOIs can be aligned to a 3×3 scout grid
    and traced back to simple `aoi-grid` counts.

We select only cells with at least 2 parking polygons.
"""

from imagery_seg.recipe import AOI, Recipe, TrainingConfig

_W, _S, _E, _N = 139.489636, 35.631850, 139.504601, 35.644067
_DLON = (_E - _W) / 3
_DLAT = (_N - _S) / 3


def _cell(i: int, j: int) -> tuple[float, float, float, float]:
    return (_W + i * _DLON, _S + j * _DLAT, _W + (i + 1) * _DLON, _S + (j + 1) * _DLAT)


recipe = Recipe.from_spec(
    name="hotosm_parking_inagi",
    imagery="hotosm",
    ground_truth="overpass",
    feature="parking",
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
            bbox=_cell(1, 0),
            region="japan",
            phenotype="japan_suburban_parking",
            notes="c1r0: south-central cell; 2 parking polygons near the lower retail/park edge",
            source_item="617e9d7d8ebcb9000515c520",
            imagery_year=2021,
        ),
        AOI(
            bbox=_cell(1, 1),
            region="japan",
            phenotype="japan_suburban_parking",
            notes="c1r1: central cell; 3 parking polygons in mixed suburban fabric",
            source_item="617e9d7d8ebcb9000515c520",
            imagery_year=2021,
        ),
        AOI(
            bbox=_cell(2, 1),
            region="japan",
            phenotype="japan_suburban_parking",
            notes="c2r1: east-central cell; 5 parking polygons, densest train parking cluster",
            source_item="617e9d7d8ebcb9000515c520",
            imagery_year=2021,
        ),
        AOI(
            bbox=_cell(0, 2),
            region="japan",
            phenotype="japan_suburban_parking",
            notes="c0r2: north-west holdout; 2 parking polygons near 城山公園 side",
            source_item="617e9d7d8ebcb9000515c520",
            imagery_year=2021,
            holdout=True,
        ),
    ),
)
