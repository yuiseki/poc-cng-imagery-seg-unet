"""Recipe: Sagamihara + Yugawara combined building training.

Merges the train AOIs from:
  - hotosm_buildings_sagamihara.py  (3 AOIs, flat suburban Kanagawa)
  - hotosm_buildings_yugawara.py    (3 AOIs, coastal hillside Shizuoka)

Both regions' original holdouts are kept as val (holdout=True).
This gives:
  train: 6 AOIs — sagamihara × 3, yugawara × 3  (~1433 + 311 = ~1744 buildings)
  val:   2 AOIs — sagamihara c3r1 (175 bldg) + yugawara SE-ne (153 bldg)

The key question: does mixing two morphologically distinct building
styles (modern flat-suburban vs traditional hillside) produce a model
that generalises better across unseen Japanese cities, or does the
domain mixing hurt within-region performance relative to the
single-region baselines?

Compare against:
  imagery-seg eval recipes/hotosm_buildings_xregion_eval.py \\
      --checkpoint tmp/runs/hotosm_buildings_sagamihara/best.pt
  imagery-seg eval recipes/hotosm_buildings_xregion_eval.py \\
      --checkpoint tmp/runs/hotosm_buildings_yugawara/best.pt
  imagery-seg eval recipes/hotosm_buildings_xregion_eval.py \\
      --checkpoint tmp/runs/hotosm_buildings_sag_yug/best.pt
"""

from imagery_seg.recipe import AOI, Recipe, TrainingConfig

recipe = Recipe.from_spec(
    name="hotosm_buildings_sag_yug",
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
    ),
    aois=(
        # ── Sagamihara train (3 AOIs) ─────────────────────────────────────
        AOI(
            bbox=(139.30296, 35.60204, 139.30819, 35.60727),
            region="sagamihara",
            notes="sag c4r0 — 421 buildings, dense suburban housing south middle-east",
        ),
        AOI(
            bbox=(139.30819, 35.60204, 139.31342, 35.60727),
            region="sagamihara",
            notes="sag c5r0 — 509 buildings, densest sub-window south-east",
        ),
        AOI(
            bbox=(139.31342, 35.60204, 139.31865, 35.60727),
            region="sagamihara",
            notes="sag c6r0 — 503 buildings, south-east edge",
        ),
        # ── Sagamihara val holdout ────────────────────────────────────────
        AOI(
            bbox=(139.29773, 35.60727, 139.30296, 35.61249),
            region="sagamihara",
            notes="sag c3r1 — 175 buildings, val holdout",
            holdout=True,
        ),
        # ── Yugawara train (3 AOIs) ───────────────────────────────────────
        AOI(
            bbox=(139.020672, 35.184826, 139.025872, 35.187773),
            region="yugawara",
            notes="yug SE-sw — 47 buildings, sparser south-west",
        ),
        AOI(
            bbox=(139.025872, 35.184826, 139.031072, 35.187773),
            region="yugawara",
            notes="yug SE-se — 58 buildings, sparser south-east",
        ),
        AOI(
            bbox=(139.020672, 35.187773, 139.025872, 35.190720),
            region="yugawara",
            notes="yug SE-nw — 206 buildings, densest sub-window",
        ),
        # ── Yugawara val holdout ──────────────────────────────────────────
        AOI(
            bbox=(139.025872, 35.187773, 139.031072, 35.190720),
            region="yugawara",
            notes="yug SE-ne — 153 buildings, val holdout",
            holdout=True,
        ),
    ),
)
