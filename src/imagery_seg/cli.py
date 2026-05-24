"""`imagery-seg` CLI.

Three subcommands:

  imagery-seg inspect RECIPE     # print the resolved Recipe summary
  imagery-seg train RECIPE       # run train_one_epoch x recipe.training.epochs
  imagery-seg eval RECIPE        # compute IoU / F1 across the recipe AOIs
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import typer

import json
from dataclasses import replace as dc_replace

from .aoi_cache import DEFAULT_CACHE_ROOT, AOICache
from .dataset import RecipeDataset
from .eval import (
    DEFAULT_AREA_FILTERS_M2,
    DEFAULT_THRESHOLDS,
    aggregate_by_region,
    evaluate,
    evaluate_per_region,
    evaluate_with_sweep,
)
from .model import build_unet
from .recipe import Recipe, load_recipe
from .train import train as run_training

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def inspect(
    recipe_path: Path = typer.Argument(..., help="Path to a recipe .py file"),
) -> None:
    """Print a one-screen summary of the resolved recipe."""
    r = load_recipe(recipe_path)
    typer.echo(f"recipe: {r.name}")
    typer.echo(f"  imagery       = {r.imagery.name}")
    typer.echo(f"  ground_truth  = {r.ground_truth.name}")
    typer.echo(f"  feature       = {r.feature.name}  (erode_px={r.feature.erode_px})")
    typer.echo(f"  cache_namespace = {r.cache_namespace}")
    typer.echo(
        f"  training: epochs={r.training.epochs} "
        f"batch_size={r.training.batch_size} lr={r.training.lr} "
        f"encoder={r.training.encoder} weights={r.training.encoder_weights} "
        f"max_side={r.training.max_side}"
    )
    typer.echo(
        f"            augment={r.training.augment} "
        f"color_jitter={r.training.aug_color_jitter} "
        f"seed={r.training.seed}"
    )
    train_aois = r.effective_train_aois
    val_aois = r.effective_val_aois
    typer.echo(f"  run_dir       = {r.run_dir}")
    typer.echo(f"  cache_root    = {DEFAULT_CACHE_ROOT} "
               f"(imagery_ns={r.imagery_cache_namespace}, "
               f"polygon_ns={r.polygon_cache_namespace})")
    typer.echo(f"  regions       = {list(r.regions)}  val_regions = {list(r.val_regions)}")
    typer.echo(f"train AOIs ({len(train_aois)}):")
    for i, aoi in enumerate(train_aois):
        suffix = f"  // {aoi.notes}" if aoi.notes else ""
        typer.echo(f"  [{i}] region={aoi.region!r} bbox={aoi.bbox}{suffix}")
    typer.echo(f"val AOIs ({len(val_aois)}):")
    for i, aoi in enumerate(val_aois):
        suffix = f"  // {aoi.notes}" if aoi.notes else ""
        typer.echo(f"  [{i}] region={aoi.region!r} bbox={aoi.bbox}{suffix}")


def _replace_aois(r: Recipe, aois) -> Recipe:
    """Return a copy of `r` with `aois` replaced and val_regions cleared.

    Frozen-dataclass safe. Used to build the train- and val-only datasets
    out of one Recipe (the source split was already applied — the child
    recipe shouldn't re-split).
    """
    return dc_replace(r, aois=tuple(aois), val_regions=())


def _build_datasets(
    r: Recipe,
    cache: AOICache | None = None,
    augment: bool = False,
    color_jitter: bool = True,
) -> tuple[RecipeDataset, RecipeDataset]:
    """Build (train_ds, val_ds) using recipe.effective_{train,val}_aois.

    Returns a val dataset with len()==0 when the recipe has no val AOIs;
    callers can detect that and skip val-time evaluation.

    augment applies to train only — val is always deterministic so the
    val IoU history is comparable across runs.
    """
    train_recipe = _replace_aois(r, r.effective_train_aois)
    val_recipe = _replace_aois(r, r.effective_val_aois)
    kwargs = dict(max_side=r.training.max_side, dst_crs="EPSG:3857", cache=cache)
    return (
        RecipeDataset(train_recipe, augment=augment, color_jitter=color_jitter, **kwargs),
        RecipeDataset(val_recipe, augment=False, **kwargs),
    )


def _build_cache(no_cache: bool) -> AOICache | None:
    return None if no_cache else AOICache(root=DEFAULT_CACHE_ROOT)


def _resolve_run_dir(r: Recipe, variant: str) -> Path:
    """Per-variant sub-directory under the recipe's run_dir.

    `--variant=baseline` writes to {output_dir}/{name}/baseline/, useful
    for A/B comparisons where two runs share a recipe but differ on
    a single knob (augmentation, lr, ...).
    """
    return (r.run_dir / variant) if variant else r.run_dir


@app.command(name="train")
def train(
    recipe_path: Path = typer.Argument(..., help="Path to a recipe .py file"),
    device: str = typer.Option("cpu", help="cpu / cuda / mps"),
    no_save: bool = typer.Option(False, "--no-save", help="Skip checkpoint writing"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Bypass the AOI cache"),
    keep_epochs: bool = typer.Option(
        False, "--keep-epochs",
        help="Also write every epoch_NNN.pt (default: best.pt + history.json only)",
    ),
    augment: bool | None = typer.Option(
        None, "--augment/--no-augment",
        help="Override TrainingConfig.augment for this run",
    ),
    color_jitter: bool | None = typer.Option(
        None, "--color-jitter/--no-color-jitter",
        help="Override TrainingConfig.aug_color_jitter (only meaningful when augment is on)",
    ),
    seed: int | None = typer.Option(
        None, "--seed", help="Override TrainingConfig.seed for this run",
    ),
    epochs: int | None = typer.Option(
        None, "--epochs", help="Override TrainingConfig.epochs for this run",
    ),
    variant: str = typer.Option(
        "", "--variant", help="Sub-directory under run_dir (for A/B comparisons)",
    ),
) -> None:
    """Run epochs of training with per-epoch val + checkpoint saving.

    Writes `{recipe.training.output_dir}/{recipe.name}/[{variant}/]best.pt`
    and `history.json` unless --no-save. Add --keep-epochs to also keep
    per-epoch checkpoints.
    """
    logging.basicConfig(level=logging.INFO)
    r = load_recipe(recipe_path)
    effective_augment = r.training.augment if augment is None else augment
    effective_color_jitter = r.training.aug_color_jitter if color_jitter is None else color_jitter
    effective_seed = r.training.seed if seed is None else seed
    effective_epochs = r.training.epochs if epochs is None else epochs
    torch.manual_seed(effective_seed)
    train_ds, val_ds = _build_datasets(
        r, cache=_build_cache(no_cache),
        augment=effective_augment,
        color_jitter=effective_color_jitter,
    )
    model = build_unet(
        encoder=r.training.encoder,
        encoder_weights=r.training.encoder_weights,
        classes=2,
    )
    optim = torch.optim.Adam(model.parameters(), lr=r.training.lr)
    run_dir = None if no_save else _resolve_run_dir(r, variant)
    result = run_training(
        model, train_ds, optim,
        epochs=effective_epochs,
        val_dataset=val_ds if len(val_ds) > 0 else None,
        batch_size=r.training.batch_size,
        device=device,
        run_dir=run_dir,
        model_meta={
            "encoder": r.training.encoder,
            "encoder_weights": r.training.encoder_weights,
            "classes": 2,
            "recipe_name": r.name,
            "augment": effective_augment,
            "color_jitter": effective_color_jitter,
            "seed": effective_seed,
            "epochs": effective_epochs,
            "variant": variant,
        },
        keep_epochs=keep_epochs,
    )
    if run_dir is not None:
        typer.echo(f"artifacts: {run_dir}/")
    for entry in result["history"]:
        typer.echo(
            f"epoch {entry['epoch']:3d}: "
            f"loss={entry['train_loss_mean']:.4f} "
            f"val_iou={entry['val_iou']} val_f1={entry['val_f1']}"
        )
    typer.echo(
        f"best epoch = {result['best']['epoch']} "
        f"(val_iou={result['best']['val_iou']:.4f})"
    )


@app.command(name="eval")
def evaluate_cmd(
    recipe_path: Path = typer.Argument(..., help="Path to a recipe .py file"),
    device: str = typer.Option("cpu", help="cpu / cuda / mps"),
    checkpoint: Path | None = typer.Option(
        None, "--checkpoint", help="Path to a .pt to load (default: {run_dir}/[{variant}/]best.pt)",
    ),
    no_cache: bool = typer.Option(False, "--no-cache", help="Bypass the AOI cache"),
    variant: str = typer.Option("", "--variant", help="Variant sub-directory to read best.pt from"),
) -> None:
    """Compute IoU + F1 on the recipe's val AOIs (or all AOIs if no split)."""
    logging.basicConfig(level=logging.INFO)
    r = load_recipe(recipe_path)
    eval_aois = r.effective_val_aois or r.aois
    eval_recipe = _replace_aois(r, eval_aois)
    ds = RecipeDataset(
        eval_recipe,
        max_side=r.training.max_side,
        dst_crs="EPSG:3857",
        cache=_build_cache(no_cache),
    )
    model = build_unet(
        encoder=r.training.encoder,
        encoder_weights=r.training.encoder_weights,
        classes=2,
    )

    ckpt_path = checkpoint or (_resolve_run_dir(r, variant) / "best.pt")
    if ckpt_path.is_file():
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["state_dict"])
        typer.echo(f"loaded checkpoint: {ckpt_path}")
    else:
        typer.echo(f"WARNING: {ckpt_path} not found; evaluating untrained model")

    metrics = evaluate(model, ds, device=device)
    typer.echo(f"IoU = {metrics['iou']:.4f}")
    typer.echo(f"F1  = {metrics['f1']:.4f}")
    region_stats = evaluate_per_region(model, ds, device=device)
    if len(region_stats) > 1 or (region_stats and next(iter(region_stats)) != ""):
        typer.echo("per-region (argmax):")
        for region, s in sorted(region_stats.items()):
            tag = region or "<untagged>"
            typer.echo(
                f"  {tag:>14}: n={s['n']} mean_iou={s['mean_iou']:.4f} "
                f"median_iou={s['median_iou']:.4f} min_iou={s['min_iou']:.4f} "
                f"mean_f1={s['mean_f1']:.4f}"
            )


def _load_model_for_recipe(
    r: Recipe,
    checkpoint: Path | None,
    device: str,
    variant: str = "",
) -> torch.nn.Module:
    """Shared helper for eval/sweep: build model, load ckpt if present."""
    model = build_unet(
        encoder=r.training.encoder,
        encoder_weights=r.training.encoder_weights,
        classes=2,
    )
    ckpt_path = checkpoint or (_resolve_run_dir(r, variant) / "best.pt")
    if ckpt_path.is_file():
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["state_dict"])
        typer.echo(f"loaded checkpoint: {ckpt_path}")
    else:
        typer.echo(f"WARNING: {ckpt_path} not found; using untrained model")
    return model


@app.command()
def sweep(
    recipe_path: Path = typer.Argument(..., help="Path to a recipe .py file"),
    device: str = typer.Option("cpu", help="cpu / cuda / mps"),
    checkpoint: Path | None = typer.Option(
        None, "--checkpoint",
        help="Path to a .pt to load (default: {run_dir}/best.pt if present)",
    ),
    on_train: bool = typer.Option(
        False, "--on-train",
        help="Sweep on the recipe's train AOIs instead of val (debugging / fit check)",
    ),
    no_cache: bool = typer.Option(False, "--no-cache", help="Bypass the AOI cache"),
    variant: str = typer.Option("", "--variant", help="Variant sub-directory to read best.pt from"),
) -> None:
    """Per-AOI threshold x area_filter sweep, persisted to thresholds.json.

    Writes {run_dir}/thresholds.json with one entry per AOI containing
    the full sweep grid and the best (threshold, area_filter_m2) pair.
    """
    logging.basicConfig(level=logging.INFO)
    r = load_recipe(recipe_path)
    aois = r.effective_train_aois if on_train else (r.effective_val_aois or r.aois)
    swept_recipe = _replace_aois(r, aois)
    ds = RecipeDataset(
        swept_recipe,
        max_side=r.training.max_side,
        dst_crs="EPSG:3857",
        cache=_build_cache(no_cache),
    )
    model = _load_model_for_recipe(r, checkpoint, device, variant=variant)

    results = evaluate_with_sweep(
        model, ds,
        thresholds=DEFAULT_THRESHOLDS,
        area_filters_m2=DEFAULT_AREA_FILTERS_M2,
        device=device,
    )

    run_dir = _resolve_run_dir(r, variant)
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / ("thresholds_train.json" if on_train else "thresholds.json")
    out_path.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    typer.echo(f"wrote {out_path}")
    for e in results:
        region_tag = (e.get("region") or "<untagged>")
        typer.echo(
            f"AOI {e['aoi_index']} [{region_tag}]: best thr={e['best_threshold']:.2f} "
            f"area_filter={e['best_area_filter_m2']:.0f}m² "
            f"IoU={e['best_iou']:.4f} F1={e['best_f1']:.4f}"
        )
    region_summary = aggregate_by_region(results, metric="best_iou")
    if len(region_summary) > 1 or (region_summary and next(iter(region_summary)) != ""):
        typer.echo("per-region (best_iou):")
        for region, s in sorted(region_summary.items()):
            tag = region or "<untagged>"
            typer.echo(
                f"  {tag:>14}: n={s['n']} mean={s['mean']:.4f} "
                f"median={s['median']:.4f} min={s['min']:.4f} "
                f"p25={s['p25']:.4f} p75={s['p75']:.4f}"
            )


@app.command()
def infer(
    recipe_path: Path = typer.Argument(..., help="Path to a recipe .py file"),
    aoi: int = typer.Option(0, "--aoi", help="AOI index into the val list (default) or full aois list (--all)"),
    use_all: bool = typer.Option(
        False, "--all", help="Index into recipe.aois rather than effective_val_aois",
    ),
    checkpoint: Path | None = typer.Option(
        None, "--checkpoint", help="Override the variant's best.pt",
    ),
    variant: str = typer.Option("", "--variant", help="Variant sub-directory"),
    threshold: float = typer.Option(0.5, "--threshold", help="Probability threshold for foreground"),
    area_filter_m2: float = typer.Option(0.0, "--area-filter-m2", help="Drop predicted components smaller than this"),
    no_cache: bool = typer.Option(False, "--no-cache"),
    device: str = typer.Option("cpu", "--device"),
) -> None:
    """Inference + visual diagnostics on a single AOI.

    Writes a 2x2 PNG (imagery / GT / pred / confusion), the prediction as
    WGS84 GeoJSON, and a small metadata.json with IoU/F1 + polygon counts.
    Use --variant=NAME to pick a non-default checkpoint.
    """
    from .infer import run_inference

    logging.basicConfig(level=logging.INFO)
    r = load_recipe(recipe_path)
    aois_pool = r.aois if use_all else (r.effective_val_aois or r.aois)
    if aoi < 0 or aoi >= len(aois_pool):
        raise typer.BadParameter(f"--aoi must be in [0, {len(aois_pool)})")
    target_aoi = aois_pool[aoi]
    target_recipe = dc_replace(r, aois=(target_aoi,), val_regions=())

    model = _load_model_for_recipe(r, checkpoint, device, variant=variant)

    out_dir = _resolve_run_dir(r, variant) / "infer" / f"aoi_{aoi}"
    meta = run_inference(
        model, target_recipe, aoi_index=0,
        threshold=threshold,
        area_filter_m2=area_filter_m2,
        out_dir=out_dir,
        cache=_build_cache(no_cache),
        device=device,
    )
    typer.echo(f"AOI {aoi} [{target_aoi.region}] bbox={target_aoi.bbox}")
    if target_aoi.notes:
        typer.echo(f"  notes: {target_aoi.notes}")
    typer.echo(f"  IoU={meta['iou']:.4f}  F1={meta['f1']:.4f}")
    typer.echo(f"  n_gt_pixels={meta['n_gt_pixels']}  n_pred_pixels={meta['n_pred_pixels']}")
    typer.echo(f"  n_pred_polys={meta['n_pred_polys']}")
    typer.echo(f"  artifacts: {out_dir}/")


def _parse_bbox(s: str) -> tuple[float, float, float, float]:
    parts = s.split(",")
    if len(parts) != 4:
        raise typer.BadParameter(f"--bbox must be W,S,E,N (got {s!r})")
    try:
        return tuple(float(p) for p in parts)  # type: ignore[return-value]
    except ValueError as e:
        raise typer.BadParameter(f"--bbox parse error: {e}") from e


@app.command(name="aoi-check")
def aoi_check_cmd(
    bbox: str = typer.Option(..., "--bbox", help="W,S,E,N in WGS84"),
    imagery: str = typer.Option("hotosm", "--imagery"),
    ground_truth: str = typer.Option("overpass", "--ground-truth"),
    feature: str = typer.Option("building", "--feature"),
    max_side: int = typer.Option(1024, "--max-side"),
    no_cache: bool = typer.Option(False, "--no-cache"),
    min_count: int | None = typer.Option(
        None, "--min-count",
        help="Override feature.min_positive_polygon_count",
    ),
    min_pixel_fraction: float | None = typer.Option(
        None, "--min-pixel-fraction",
        help="Override feature.min_positive_pixel_fraction",
    ),
    min_valid: float = typer.Option(0.50, "--min-valid"),
) -> None:
    """Score one AOI against the Phase B selection criteria.

    Prints an AOICheck json. Useful for vetting a bbox before adding it
    to a phenotype catalog.
    """
    from .aoi_check import check_aoi
    from .features import get_feature
    from .ground_truth import get_source as get_gt
    from .imagery import get_source as get_imagery

    logging.basicConfig(level=logging.INFO)
    b = _parse_bbox(bbox)
    img_src = get_imagery(imagery)
    gt_src = get_gt(ground_truth)
    feat = get_feature(feature)

    check = check_aoi(
        b, img_src, gt_src, feat,
        max_side=max_side,
        min_polygon_count=min_count,
        min_pixel_fraction=min_pixel_fraction,
        min_valid_fraction=min_valid,
    )
    typer.echo(json.dumps(check.to_dict(), indent=2))


@app.command(name="aoi-grid")
def aoi_grid_cmd(
    bbox: str = typer.Option(..., "--bbox", help="W,S,E,N in WGS84"),
    cols: int = typer.Option(7, "--cols"),
    rows: int = typer.Option(4, "--rows"),
    imagery: str = typer.Option("hotosm", "--imagery"),
    ground_truth: str = typer.Option("overpass", "--ground-truth"),
    feature: str = typer.Option("building", "--feature"),
    max_side: int = typer.Option(1024, "--max-side"),
    min_count: int | None = typer.Option(
        None, "--min-count", help="Override feature.min_positive_polygon_count",
    ),
    min_pixel_fraction: float | None = typer.Option(
        None, "--min-pixel-fraction",
        help="Override feature.min_positive_pixel_fraction",
    ),
    min_valid: float = typer.Option(0.50, "--min-valid"),
    out: Path | None = typer.Option(None, "--out", help="Write JSON array of AOICheck"),
) -> None:
    """Tile a parent bbox into rows × cols and score every cell.

    Use to scout new HOTOSM COGs for usable sub-AOIs. A short summary
    table goes to stdout; full results go to --out (json) when given.
    """
    from .aoi_check import check_aoi
    from .features import get_feature
    from .ground_truth import get_source as get_gt
    from .imagery import get_source as get_imagery

    logging.basicConfig(level=logging.INFO)
    W, S, E, N = _parse_bbox(bbox)
    img_src = get_imagery(imagery)
    gt_src = get_gt(ground_truth)
    feat = get_feature(feature)

    dlon = (E - W) / cols
    dlat = (N - S) / rows
    results: list[dict] = []
    typer.echo(
        f"{'cell':>6}  {'polys':>6}  {'pix%':>6}  {'valid':>6}  "
        f"{'quality':>8}  {'pass':>5}  notes"
    )
    typer.echo("-" * 80)
    for j in range(rows):
        for i in range(cols):
            w = W + i * dlon
            e = w + dlon
            s = S + j * dlat
            n = s + dlat
            cell_bbox = (w, s, e, n)
            check = check_aoi(
                cell_bbox, img_src, gt_src, feat,
                max_side=max_side,
                min_polygon_count=min_count,
                min_pixel_fraction=min_pixel_fraction,
                min_valid_fraction=min_valid,
            )
            notes = "; ".join(check.reasons[:1] + check.warnings[:1])[:40]
            typer.echo(
                f"c{i}r{j:<2}  {check.positive_polygon_count:>6}  "
                f"{100*check.positive_pixel_fraction:>5.2f}%  "
                f"{check.valid_pixel_fraction:>6.3f}  "
                f"{check.quality_score:>8.3f}  "
                f"{'YES' if check.hard_pass else 'no ':>5}  {notes}"
            )
            results.append(check.to_dict())

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        typer.echo(f"wrote {out}")


@app.command(name="validate")
def validate_cmd(
    recipe_path: Path = typer.Argument(..., help="Path to a recipe .py file"),
) -> None:
    """Recipe-level Phase B checks: phenotype consistency + train/val
    spatial separation. Returns nonzero exit code on errors."""
    from .aoi_check import validate_recipe
    r = load_recipe(recipe_path)
    rep = validate_recipe(r)
    if rep["errors"]:
        typer.echo("ERRORS:")
        for e in rep["errors"]:
            typer.echo(f"  - {e}")
    if rep["warnings"]:
        typer.echo("WARNINGS:")
        for w in rep["warnings"]:
            typer.echo(f"  - {w}")
    if not rep["errors"] and not rep["warnings"]:
        typer.echo("OK — all Phase B recipe-level checks pass.")
    if rep["errors"]:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
