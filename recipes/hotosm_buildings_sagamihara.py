"""Recipe: HOTOSM Sagamihara — DRONEBIRD 2019 peacetime imagery, 4 AOIs.

Companion to `hotosm_buildings_yugawara.py`. Built to test the hypothesis:

  > A model trained on Yugawara generalises poorly to Sagamihara
  > because the building appearance is different — Sagamihara is
  > suburban-Tokyo housing (newer, more standardised roofs), Yugawara
  > is a coastal hot-spring town (older, traditional Japanese roofs).

The yugawara-trained best.pt (val IoU 0.4367 on Yugawara holdout via sweep)
hit only IoU 0.11 on Sagamihara c5r0 — but until we train a model
*on* Sagamihara, we don't know whether the gap is "Sagamihara is hard"
or "Yugawara features don't transfer". This recipe answers the latter:
if Sagamihara-only training reaches comparable val IoU to Yugawara's,
the data is learnable and the cross-city gap is genuine. If it stays
low, something about this imagery / OSM combination is unusually hard.

Source COG: HOTOSM item 5dae963873c69f000530ee74 (DRONEBIRD, Aihara,
Sagamihara, 2019-10-20, GSD ~5cm). CC-BY-4.0. ~3.3 x 2.3 km tile with
~2700 OSM buildings total; we use 4 of the dense south-row sub-cells.

train (3 AOIs, ~1433 buildings total):
  c4r0  421 buildings  — south middle-east, dense suburban housing
  c5r0  509 buildings  — south east, densest of the cluster
  c6r0  503 buildings  — south easternmost edge

val (1 AOI, 175 buildings, holdout):
  c3r1  175 buildings  — one row north of c4r0, spatially separated
                         from train cluster while still capturing the
                         same suburban Sagamihara building style
"""

from imagery_seg.recipe import AOI, Recipe, TrainingConfig

recipe = Recipe.from_spec(
    name="hotosm_buildings_sagamihara",
    imagery="hotosm",
    ground_truth="overpass",
    feature="building",
    training=TrainingConfig(
        epochs=50,
        batch_size=2,
        lr=1e-4,
        encoder="resnet34",
        encoder_weights="imagenet",
        max_side=1024,
        # augment / aug_color_jitter / seed inherit defaults
        # (augment=True, color_jitter=True, seed=0) — same as Yugawara
        # so the only variable across the two experiments is the data.
    ),
    aois=(
        AOI(
            bbox=(139.30296, 35.60204, 139.30819, 35.60727),
            region="japan",
            notes="c4r0, 421 buildings, dense suburban housing south middle-east",
        ),
        AOI(
            bbox=(139.30819, 35.60204, 139.31342, 35.60727),
            region="japan",
            notes="c5r0, 509 buildings, densest sub-window south-east",
        ),
        AOI(
            bbox=(139.31342, 35.60204, 139.31865, 35.60727),
            region="japan",
            notes="c6r0, 503 buildings, south-east edge",
        ),
        AOI(
            bbox=(139.29773, 35.60727, 139.30296, 35.61249),
            region="japan",
            notes=(
                "c3r1, 175 buildings, val holdout — one row north of train cluster, "
                "spatially separated but same Sagamihara suburban building style"
            ),
            holdout=True,
        ),
    ),
)
