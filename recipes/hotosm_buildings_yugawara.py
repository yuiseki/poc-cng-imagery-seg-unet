"""Recipe: HOTOSM Yugawara — 4 sub-windows of one COG, 3 train / 1 val.

Within-region replacement for `hotosm_buildings_atami_manila.py`. All 4
AOIs come from a single HOTOSM item (id=5f87c60b8d05870005b91bbf,
2017-03-06, ~1.9km x 1.3km tile around Yugawara, Shizuoka). The dense
SE quadrant of the tile was further split 2x2 to produce four ~473x327m
sub-windows, each with enough buildings for a non-trivial mask.

train (3 AOIs, 311 buildings total):
  - SE-sw  47 buildings  - sparser south-west
  - SE-se  58 buildings  - sparser south-east
  - SE-nw 206 buildings  - densest

val (1 AOI, 153 buildings, holdout=True):
  - SE-ne 153 buildings  - chosen as val: diagonally opposite from the
                          densest train AOI (SE-nw) so it's the most
                          spatially separated of the four, and large
                          enough to give a clean IoU signal.

Why this is the right A/B target (not Atami train / Manila val): the
training axis (augmentation, loss, lr) is what we want to measure, not
the cross-region gap that dominated the legacy recipe.
"""

from imagery_seg.recipe import AOI, Recipe, TrainingConfig

recipe = Recipe.from_spec(
    name="hotosm_buildings_yugawara",
    imagery="hotosm",
    ground_truth="overpass",
    feature="building",
    training=TrainingConfig(
        # 30-epoch A/B (Yugawara, seed=0) showed val IoU still climbing at
        # epoch 30 for baseline and aug_full peaking at epoch 28. Run to 50
        # so the curve has room to plateau or overfit and we can read where
        # the real ceiling sits before doing more data work.
        epochs=50,
        batch_size=2,
        lr=1e-4,
        encoder="resnet34",
        encoder_weights="imagenet",
        max_side=1024,
        # augment / aug_color_jitter / seed all inherit the new defaults
        # (augment=True, color_jitter=True, seed=0).
    ),
    aois=(
        AOI(
            bbox=(139.020672, 35.184826, 139.025872, 35.187773),
            region="japan",
            notes="SE-sw, 47 buildings, sparser south-west of the dense block",
        ),
        AOI(
            bbox=(139.025872, 35.184826, 139.031072, 35.187773),
            region="japan",
            notes="SE-se, 58 buildings, sparser south-east",
        ),
        AOI(
            bbox=(139.020672, 35.187773, 139.025872, 35.190720),
            region="japan",
            notes="SE-nw, 206 buildings, densest sub-window",
        ),
        AOI(
            bbox=(139.025872, 35.187773, 139.031072, 35.190720),
            region="japan",
            notes="SE-ne, 153 buildings, val holdout (most spatially separated from SE-nw)",
            holdout=True,
        ),
    ),
)
