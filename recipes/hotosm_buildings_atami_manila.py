"""Recipe (legacy / research record): Atami train, Manila val.

Kept as a record of the very first end-to-end run on real HOTOSM imagery
(see `tmp/runs/hotosm_buildings_atami_manila/{history.json, best.pt}`).

This recipe trains on 1 Japan AOI and evaluates on 1 Philippines AOI,
which conflates "augmentation / loss / lr improvements" with the
"cross-region generalisation gap". Don't use it for actual model
development — `hotosm_buildings_yugawara.py` is the within-region
replacement. The Manila vs Atami numbers stay useful as a worst-case
OOD baseline.

Observed in the first run (Adam lr=1e-4, CE, no augmentation, 10 epochs):
  best epoch = 1, val IoU = 0.2373, F1 = 0.3835
  train loss kept dropping (0.64 -> 0.40) while val IoU monotonically
  declined to 0.16 — textbook overfit to Atami within a single epoch
  once the model leaves ImageNet-pretrained weights behind.
"""

from imagery_seg.recipe import AOI, Recipe, TrainingConfig

recipe = Recipe.from_spec(
    name="hotosm_buildings_atami_manila",
    imagery="hotosm",
    ground_truth="overpass",
    feature="building",
    training=TrainingConfig(
        epochs=10,
        batch_size=2,
        lr=1e-4,
        encoder="resnet34",
        encoder_weights="imagenet",
        max_side=1024,
    ),
    aois=(
        AOI(
            bbox=(139.075, 35.113, 139.0815, 35.118),
            region="japan",
            notes="Atami dense block south-east of the COG center "
                  "(item 60e5afbe..., ~235 buildings).",
        ),
        AOI(
            bbox=(120.968, 14.587, 120.980, 14.592),
            region="philippines",
            notes="Manila dense urban block, HOTOSM imagery present.",
        ),
    ),
    val_regions=("philippines",),
)
