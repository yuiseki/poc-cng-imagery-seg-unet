# poc-cng-imagery-seg-unet

> **任意の衛星/空撮 imagery × 任意の OSM 由来 ground truth × 任意の地物カテゴリ** を入力に U-Net セグメンテーションモデルを学習するパイプラインの scaffold。

## 世界観

```
                     ┌────────────────┐
   bbox / AOI list ──▶│  ImagerySource │── COG / tile bytes ──┐
                     └────────────────┘                       │
                                                              ▼
                     ┌────────────────┐               ┌──────────────┐
   bbox + feature ──▶│GroundTruthSrc  │── polygons ──▶│  Rasterizer  │── mask ──┐
                     └────────────────┘               └──────────────┘          │
                                                                                ▼
                                                                  ┌──────────────────┐
                                                                  │  (image, mask)    │
                                                                  │   PyTorch Dataset │
                                                                  └────────┬─────────┘
                                                                           │
                                                                           ▼
                                                                  ┌──────────────────┐
                                                                  │   smp.Unet train │
                                                                  └────────┬─────────┘
                                                                           ▼
                                                                  ┌──────────────────┐
                                                                  │ checkpoint + F1  │
                                                                  └──────────────────┘
```

3 つを pluggable に:

- **ImagerySource** — どこから画像を取るか
  - `HotosmImagery` (HOTOSM STAC + COG via `/vsicurl/`) ← 最初の target
  - `Sentinel2Imagery` (Microsoft Planetary Computer STAC + Sentinel-2 COG) ← stub
- **GroundTruthSource** — どこから OSM ベクタを取るか
  - `OverpassGT` (`overpass.yuiseki.net` ← self-hosted、 rate-limit から自由)
  - `VectorTileGT` (`tile.yuiseki.net` の vector tile) ← stub
- **FeatureSpec** — 何を抽出するか
  - `BuildingFeature` (`way["building"]`) ← 最初の target
  - `ParkFeature` (`way["leisure"="park"]` ほか) ← stub

これらの 3 軸を `Recipe` (dataclass) で組み合わせる。 `recipes/hotosm_buildings.py` が initial reference、 `recipes/sentinel2_parks.py` が「次の自然な発展」の sketch。

## 状態

scaffold のみ。 学習精度は未追求。 各層に最低限の動作確認テストと smoke 経路がある状態を作って、 ここから model / dataset の質を積み上げるための足場とする。

姉妹リポ [poc-cng-hotosm-vector-seg](../poc-cng-hotosm-vector-seg) で確立した:
- COG /vsicurl/ + reproject パターン
- AOI window cache + atomic write + file lock
- smp.Unet (resnet34 / ImageNet) + OSM mask 訓練
- per-AOI threshold calibration

をそのまま **HOTOSM imagery × Building** という 1 組み合わせ向けの specialised pipeline と見なし、 本リポではそれを **N imagery × M feature** に拡張する。

## ディレクトリ

```
src/imagery_seg/
├── recipe.py             # Recipe dataclass (imagery + GT + feature + training config)
├── imagery/
│   ├── base.py           # ImagerySource ABC
│   ├── hotosm.py         # HOTOSM STAC + COG (production-ready 移植)
│   └── sentinel2.py      # MPC Sentinel-2 (stub)
├── ground_truth/
│   ├── base.py           # GroundTruthSource ABC
│   ├── overpass.py       # overpass.yuiseki.net
│   └── vector_tile.py    # tile.yuiseki.net (stub)
├── features/
│   ├── base.py           # FeatureSpec ABC
│   ├── building.py       # way["building"]
│   └── park.py           # way["leisure"="park"] ほか (stub)
├── cache.py              # file_lock, atomic write, key 構築
├── rasterize.py          # polygons + transform -> binary mask
├── dataset.py            # torch.utils.data.Dataset wrapping (image, mask) pairs
├── model.py              # smp.Unet build / load / preprocess
├── train.py              # training loop
├── eval.py               # F1 / IoU 計測
└── cli.py                # `imagery-seg <subcommand>`

recipes/
├── hotosm_buildings.py   # 最初の target
└── sentinel2_parks.py    # 次の自然な発展 (stub)

stages/                   # 実験 / 単発実行スクリプト
└── 01_recipe_smoke.py    # recipe を 1 AOI 分動かして tmp/ に書き出すだけ

tests/                    # pytest
```

## 起動

```bash
uv sync
# scaffold が動くか確認:
uv run pytest

# AOI を 1 つ取って (image, mask) のペアを tmp/ に書き出す smoke:
uv run python stages/01_recipe_smoke.py recipes/hotosm_buildings.py

# 学習 (未実装スタブ):
uv run imagery-seg train recipes/hotosm_buildings.py
```

## non-goals (初期スコープ外)

- 高い F1 を出すこと (model 学習自体は scaffold が動いた後の課題)
- production server / Knative デプロイ
- multi-class segmentation (binary 1 class ずつから始める)
- 大規模 AOI batch 抽出パイプライン

これらは姉妹リポで一度通った道。 ここでは「複数 imagery × 複数 feature を試せる土台」を最優先する。
