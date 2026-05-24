"""Recipe: HOTOSM Isehara town hall parking — first feature=parking recipe.

Built to test parking as a third feature type after building and road:

  - building: closed polygon, count-rich (hundreds per AOI)
  - road:     LineString → buffered, count-medium (tens of ways)
  - parking:  closed polygon, **count-sparse but pixel-rich**
              (typical residential cell has 1-5 polygons, but each one
              is a large surface lot covering 10-25% of pixels)

This recipe stresses Phase B v2's feature-aware filter — with
ParkingFeature defaults of (min_count=2, min_pixel_fraction=0.02),
sparse-polygon AOIs that still have meaningful pixel coverage pass.

Source: HOTOSM STAC item 61ee8fb5ac5a1d0005830785 (DRONEBIRD /
mapconcierge, 2022-01-24, ~2.6cm GSD over Isehara city hall area,
~1.1km × 0.5km tile). The town hall + surrounding institutional
buildings cluster generates more OSM parking mapping than typical
residential blocks.

Layout (within the 6×4 grid we evaluated):

  train (3 AOIs, 17 parking polygons, ~20% pixel signal each):
    c0r3   6 polygons, 19.06% px parking
    c2r3   7 polygons, 25.96% px parking
    c1r2   4 polygons, 14.42% px parking
  val (1 AOI, 5 polygons, 22% px, spatially separated from train):
    c2r0   5 polygons, 21.84% px parking

c2r0 sits ~390m south of the densest train cell, well above the
200m MIN_TRAIN_VAL_DISTANCE_M leakage guard.
"""

from imagery_seg.recipe import AOI, Recipe, TrainingConfig

# Grid origin + cell size for traceability
_W, _S, _E, _N = 139.310839, 35.401168, 139.320643, 35.405812
_DLON = (_E - _W) / 6
_DLAT = (_N - _S) / 4


def _cell(i: int, j: int) -> tuple[float, float, float, float]:
    return (_W + i * _DLON, _S + j * _DLAT, _W + (i + 1) * _DLON, _S + (j + 1) * _DLAT)


recipe = Recipe.from_spec(
    name="hotosm_parking_isehara",
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
            bbox=_cell(0, 3),
            region="japan",
            phenotype="japan_institutional_parking",
            notes="c0r3: 6 parking polygons, 19% pixel coverage",
            source_item="61ee8fb5ac5a1d0005830785",
            imagery_year=2022,
        ),
        AOI(
            bbox=_cell(2, 3),
            region="japan",
            phenotype="japan_institutional_parking",
            notes="c2r3: 7 parking polygons, 26% pixel coverage (densest train)",
            source_item="61ee8fb5ac5a1d0005830785",
            imagery_year=2022,
        ),
        AOI(
            bbox=_cell(1, 2),
            region="japan",
            phenotype="japan_institutional_parking",
            notes="c1r2: 4 parking polygons, 14% pixel coverage, valid 100%",
            source_item="61ee8fb5ac5a1d0005830785",
            imagery_year=2022,
        ),
        AOI(
            bbox=_cell(2, 0),
            region="japan",
            phenotype="japan_institutional_parking",
            notes="c2r0: 5 parking polygons, 22% px, val holdout (~390m south of train)",
            source_item="61ee8fb5ac5a1d0005830785",
            imagery_year=2022,
            holdout=True,
        ),
    ),
)
