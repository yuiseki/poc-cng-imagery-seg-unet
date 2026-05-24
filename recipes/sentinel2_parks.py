"""Recipe: Sentinel-2 imagery (MPC) + Overpass(yuiseki) parks.

The "next natural evolution" recipe. Inspect works; train/eval will
fail loudly until Sentinel2Imagery.fetch_for_bbox is wired to SAS
signing.
"""

from imagery_seg.recipe import AOI, Recipe, TrainingConfig

recipe = Recipe.from_spec(
    name="sentinel2_parks",
    imagery="sentinel2",
    ground_truth="overpass",
    feature="park",
    training=TrainingConfig(
        epochs=20,
        batch_size=4,
        lr=2e-4,
        encoder="resnet34",
        encoder_weights="imagenet",
        max_side=512,  # Sentinel-2 10m pixels, smaller tiles per AOI
    ),
    aois=(
        AOI(
            bbox=(139.60, 35.55, 139.90, 35.80),
            region="japan",
            notes="Tokyo / Kanto plain — large parks (Yoyogi, Showa-kinen, etc).",
        ),
        AOI(
            bbox=(120.95, 14.55, 121.05, 14.62),
            region="philippines",
            notes="Manila — Rizal Park area.",
        ),
    ),
    val_regions=("philippines",),
)
