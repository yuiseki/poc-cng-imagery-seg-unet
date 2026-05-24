"""Recipe: HOTOSM Sagamihara parks — first HOTOSM-backed park recipe.

Built from the same Sagamihara / Aihara 2019 DRONEBIRD imagery family as
`hotosm_buildings_sagamihara.py`, but with feature swapped to `park`.

Why Sagamihara first:

  - the HOTOSM imagery is peacetime / training-style imagery rather than
    a post-disaster capture, so OSM-vs-imagery drift is relatively low;
  - querying the wider Sagamihara bbox against `overpass.yuiseki.net`
    returned four park-ish ways that are still visually legible in the
    ortho:
      - 相原根岸せせらぎ公園
      - 相原中央公園
      - 町屋第三公園
      - 中相原スポーツ広場

Unlike buildings, parks are sparse in this micro-area: one AOI tends to
contain one dominant polygon rather than dozens of positives. For this
recipe we therefore relax `ParkFeature.min_positive_polygon_count` from 3
to 1 while keeping the positive pixel-fraction gate as-is. The point is
to validate the `hotosm × overpass × park` path without coupling it to
the unfinished Sentinel-2 backend.
"""

from imagery_seg.recipe import AOI, Recipe, TrainingConfig

recipe = Recipe.from_spec(
    name="hotosm_parks_sagamihara",
    imagery="hotosm",
    ground_truth="overpass",
    feature="park",
    feature_kwargs={
        "min_positive_polygon_count": 1,
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
            bbox=(139.30991, 35.60399, 139.31292, 35.60495),
            region="japan",
            phenotype="japan_suburban_park",
            notes="相原根岸せせらぎ公園: elongated neighbourhood park with waterway",
            source_item="5dae963873c69f000530ee74",
            imagery_year=2019,
        ),
        AOI(
            bbox=(139.30736, 35.60175, 139.30816, 35.60245),
            region="japan",
            phenotype="japan_suburban_park",
            notes="町屋第三公園: very small pocket park; intentionally tight crop",
            source_item="5dae963873c69f000530ee74",
            imagery_year=2019,
        ),
        AOI(
            bbox=(139.31252, 35.60534, 139.31405, 35.60603),
            region="japan",
            phenotype="japan_suburban_park",
            notes="中相原スポーツ広場: open sports ground / recreation-ground style park",
            source_item="5dae963873c69f000530ee74",
            imagery_year=2019,
        ),
        AOI(
            bbox=(139.31638, 35.60790, 139.32258, 35.61420),
            region="japan",
            phenotype="japan_suburban_park",
            notes="相原中央公園: largest park in the local cluster, val holdout",
            source_item="5dae963873c69f000530ee74",
            imagery_year=2019,
            holdout=True,
        ),
    ),
)
