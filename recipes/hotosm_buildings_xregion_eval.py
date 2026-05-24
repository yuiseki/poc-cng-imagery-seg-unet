"""Recipe: Cross-region zero-shot transfer eval for building models.

All AOIs are holdout=True — no train AOIs. Use with --checkpoint:

  imagery-seg eval recipes/hotosm_buildings_xregion_eval.py \\
      --checkpoint tmp/runs/hotosm_buildings_sagamihara/best.pt

  imagery-seg eval recipes/hotosm_buildings_xregion_eval.py \\
      --checkpoint tmp/runs/hotosm_buildings_yugawara/best.pt

AOI selection criteria:
  - GSD < 0.12 m (HOTOSM drone / aerial, not Landsat)
  - HOTOSM STAC coverage confirmed
  - positive_polygon_count >= 30 (aoi-check verified)
  - Geographically distinct from existing training recipes
    (no Sagamihara, Yugawara, Isehara, Inagi, or Atami cells)

Regions covered:
  iwate     — 東北 / 岩手 / 沼宮内 : Tohoku mountain village (DRONEBIRD 2021)
  hiroshima — 中四国 / 広島県 / 似島 : Hiroshima Bay island settlement (2018)
  saga      — 九州 / 佐賀県 / 大町・北方 : flat Kyushu rural/suburban (DRONEBIRD 2019)
  nagano    — 中部 / 長野県 / 大町 : alpine mountain town (DRONEBIRD 2024)

All four regions are well outside the Kanagawa / coastal-Shizuoka corridor
where all current training recipes are anchored.
"""

from imagery_seg.recipe import AOI, Recipe, TrainingConfig

recipe = Recipe.from_spec(
    name="hotosm_buildings_xregion_eval",
    imagery="hotosm",
    ground_truth="overpass",
    feature="building",
    training=TrainingConfig(
        epochs=0,
        encoder="resnet34",
        encoder_weights="imagenet",
        max_side=1024,
    ),
    aois=(
        # ── 東北 / 岩手 ────────────────────────────────────────────────────
        AOI(
            bbox=(141.21458, 39.96248, 141.22096, 39.96893),
            region="iwate",
            notes="Numakunai, Iwate — rural Tohoku mountain village, 380 buildings, DRONEBIRD 2021",
            source_item="6190cc1a5024550007345838",
            imagery_year=2021,
            holdout=True,
        ),
        # ── 中四国 / 広島・似島 ───────────────────────────────────────────
        AOI(
            bbox=(132.43361, 34.31110, 132.44233, 34.31925),
            region="hiroshima",
            notes="Ninoshima island c2r2, Hiroshima — island settlement, 249 buildings, 2018",
            source_item="5bbdcd62b5f1fe00054e0d34",
            imagery_year=2018,
            holdout=True,
        ),
        AOI(
            bbox=(132.42489, 34.30295, 132.43361, 34.31110),
            region="hiroshima",
            notes="Ninoshima island c1r1, Hiroshima — island settlement, 112 buildings, 2018",
            source_item="5bbdcd62b5f1fe00054e0d34",
            imagery_year=2018,
            holdout=True,
        ),
        # ── 九州 / 佐賀 ──────────────────────────────────────────────────
        AOI(
            bbox=(130.07004, 33.21134, 130.07759, 33.21779),
            region="saga",
            notes="Kitagata, Saga — flat Kyushu rural, 208 buildings, DRONEBIRD 2019",
            source_item="5d78e8faf719820008245cfa",
            imagery_year=2019,
            holdout=True,
        ),
        AOI(
            bbox=(130.09615, 33.20101, 130.10582, 33.20882),
            region="saga",
            notes="Omachi, Saga — flat Kyushu rural/suburban, 46 buildings, DRONEBIRD 2019",
            source_item="5d7b91e50d28b10007763f85",
            imagery_year=2019,
            holdout=True,
        ),
        # ── 中部 / 長野 ──────────────────────────────────────────────────
        AOI(
            bbox=(137.79589, 36.50737, 137.80164, 36.51149),
            region="nagano",
            notes="Omachi, Nagano — alpine mountain town, 36 buildings, DRONEBIRD 2024",
            source_item="663a10496049ef00013b841e",
            imagery_year=2024,
            holdout=True,
        ),
    ),
)
