"""Recipe: HOTOSM Sagamihara roads — same AOIs as the buildings recipe,
feature swapped to "road".

Companion to `hotosm_roads_yugawara.py`. Built specifically to test the
hypothesis:

  > Road segmentation's low IoU on Yugawara (sweep best 0.27 vs
  > buildings' 0.44) might be specific to Yugawara's hillside terrain
  > — winding narrow streets, tree-canopy occlusion, irregular layout
  > — rather than something intrinsic to "road as a class".

Sagamihara is the flat-suburban-grid counterpoint: clear orthogonal
streets, minimal canopy, regular block size. If road training reaches
materially higher IoU on Sagamihara with the same architecture / loss /
training schedule / buffer widths, the Yugawara result was about
terrain. If road still plateaus low, lines are intrinsically harder
for the UNet+CE setup than compact "object" classes like buildings.

Same 4 AOIs as `hotosm_buildings_sagamihara.py` so the only variable
between (Yug road, Sag road, Yug bldg, Sag bldg) is the swap.
"""

from imagery_seg.recipe import AOI, Recipe, TrainingConfig

recipe = Recipe.from_spec(
    name="hotosm_roads_sagamihara",
    imagery="hotosm",
    ground_truth="overpass",
    feature="road",
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
            bbox=(139.30296, 35.60204, 139.30819, 35.60727),
            region="japan",
            phenotype="japan_suburban",
            notes="c4r0 — same AOI as Sag building recipe (suburban grid)",
        ),
        AOI(
            bbox=(139.30819, 35.60204, 139.31342, 35.60727),
            region="japan",
            phenotype="japan_suburban",
            notes="c5r0 — densest cell, regular block layout",
        ),
        AOI(
            bbox=(139.31342, 35.60204, 139.31865, 35.60727),
            region="japan",
            phenotype="japan_suburban",
            notes="c6r0 — south-east edge",
        ),
        AOI(
            bbox=(139.29773, 35.60727, 139.30296, 35.61249),
            region="japan",
            phenotype="japan_suburban",
            notes="c3r1 — val holdout (same as building recipe)",
            holdout=True,
        ),
    ),
)
