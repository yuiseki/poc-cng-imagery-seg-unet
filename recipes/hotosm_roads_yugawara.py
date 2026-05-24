"""Recipe: HOTOSM Yugawara roads — same AOIs/imagery as the buildings
recipe, but feature swapped to "road".

The point of this recipe is to exercise the N-feature axis of the
scaffold (`Recipe = imagery × ground_truth × feature`) on a feature
whose geometry is fundamentally different from buildings:

  - OSM building = closed way → Polygon (rasterise directly)
  - OSM highway  = open way   → LineString (buffer by tag-derived
                                width → Polygon, then rasterise)

If the scaffold genuinely supports plurality, the only diff between
this recipe and `hotosm_buildings_yugawara.py` should be `feature="road"`
plus a different recipe `name` (so artifacts go to their own run_dir).

The actual road geometry processing lives in `RoadFeature.to_polygons`;
the cache layer stores the post-buffered polygons under the
`overpass__road` polygon-namespace (separate from `overpass__building`).

After training, compare `tmp/runs/hotosm_roads_yugawara/aug_full_50ep/`
to `tmp/runs/hotosm_buildings_yugawara/aug_full_50ep/`. Same imagery,
same AOIs, same architecture and training schedule — the only variable
is the segmentation target. Whatever IoU difference shows up is
attributable to the feature's intrinsic difficulty + how good OSM
mapping is for it in this region.
"""

from imagery_seg.recipe import AOI, Recipe, TrainingConfig

recipe = Recipe.from_spec(
    name="hotosm_roads_yugawara",
    imagery="hotosm",
    ground_truth="overpass",
    feature="road",
    training=TrainingConfig(
        # Same config as hotosm_buildings_yugawara.py so the building/road
        # comparison is honest. If road needs different epochs / lr,
        # tune in a follow-up.
        epochs=50,
        batch_size=2,
        lr=1e-4,
        encoder="resnet34",
        encoder_weights="imagenet",
        max_side=1024,
    ),
    aois=(
        AOI(
            bbox=(139.020672, 35.184826, 139.025872, 35.187773),
            region="japan",
            phenotype="japan_suburban",
            notes="SE-sw — same AOI as building recipe (different feature axis)",
        ),
        AOI(
            bbox=(139.025872, 35.184826, 139.031072, 35.187773),
            region="japan",
            phenotype="japan_suburban",
            notes="SE-se — same AOI as building recipe",
        ),
        AOI(
            bbox=(139.020672, 35.187773, 139.025872, 35.190720),
            region="japan",
            phenotype="japan_suburban",
            notes="SE-nw — same AOI as building recipe (densest urban grid)",
        ),
        AOI(
            bbox=(139.025872, 35.187773, 139.031072, 35.190720),
            region="japan",
            phenotype="japan_suburban",
            notes="SE-ne — val holdout, same as building recipe",
            holdout=True,
        ),
    ),
)
