"""Recipe: imagery x ground-truth x feature x training knobs.

A Recipe is the unit of configuration in this project. It can be
constructed in code (see recipes/hotosm_buildings.py) or, eventually,
loaded from YAML. The training CLI takes a recipe path + a list of
AOIs and turns them into training pairs.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .features import FeatureSpec, get_feature
from .ground_truth import GroundTruthSource, get_source as get_gt_source
from .imagery import ImagerySource, get_source as get_imagery_source


Bbox = tuple[float, float, float, float]


@dataclass(frozen=True)
class AOI:
    """One area of interest with a region tag.

    `region` is the unit train/val splitting and per-region aggregation
    operate on. Use the same region string across AOIs that share
    building morphology / urban form (e.g. "japan", "philippines",
    "se_asia_dense_urban").

    `notes` is free-form: where the AOI was sourced from, why it was
    picked, known caveats. It surfaces in `imagery-seg inspect`.

    `holdout=True` marks this AOI as val regardless of region — used
    for within-region splits (multiple AOIs of one region with one
    held out). Combines with Recipe.val_regions: an AOI is val iff
    its region is in val_regions OR holdout is True.
    """

    bbox: Bbox
    region: str
    notes: str = ""
    holdout: bool = False
    #: Architectural family this AOI belongs to (e.g. "japan_suburban",
    #: "se_asia_dense_colonial"). Used by recipe-level validation: train
    #: and val AOIs in a recipe should share phenotype since the
    #: cross-phenotype generalisation gap is large.
    phenotype: str = ""
    #: STAC item id of the source COG (cache lineage / disaster lookup).
    source_item: str = ""
    #: Year of imagery capture (helps spot OSM-vs-imagery time drift).
    imagery_year: int | None = None


def _coerce_aois(aois: Iterable) -> tuple[AOI, ...]:
    """Allow recipe authors to pass either AOI(...) objects or bare
    bbox 4-tuples. Bare tuples get region="" — useful for one-off
    smoke recipes that don't yet care about regions.
    """
    result: list[AOI] = []
    for a in aois:
        if isinstance(a, AOI):
            result.append(a)
            continue
        if isinstance(a, Sequence) and len(a) == 4:
            result.append(AOI(bbox=tuple(a), region="", notes=""))  # type: ignore[arg-type]
            continue
        raise TypeError(
            f"AOI must be an AOI(...) or a 4-tuple bbox, got {type(a).__name__}"
        )
    return tuple(result)


@dataclass(frozen=True)
class TrainingConfig:
    """Training-time knobs. Kept on the Recipe so different recipes
    can ship sensible defaults (e.g. Sentinel-2 needs smaller batch,
    bigger LR than HOTOSM)."""
    epochs: int = 10
    batch_size: int = 4
    lr: float = 1e-4
    encoder: str = "resnet34"
    encoder_weights: str | None = "imagenet"
    max_side: int = 1024
    output_dir: str = "tmp/runs"
    #: Geometric (D4 symmetry) + photometric (brightness ±20%) augmentation
    #: on the train dataset. Empirically validated on the Yugawara A/B
    #: (30 epoch, seed=0): aug_full peaked val IoU 0.3073 vs 0.2575 baseline.
    augment: bool = True
    aug_color_jitter: bool = True
    seed: int = 0


@dataclass(frozen=True)
class Recipe:
    """One end-to-end training configuration.

    AOIs are first-class `AOI(bbox, region, notes)` objects. The
    train/val split is normally driven by `val_regions`: every AOI
    whose region is listed becomes val, the rest train. With
    val_regions empty and >=2 AOIs, the last AOI is implicitly the
    val one (single-AOI smoke recipes still work).
    """
    name: str
    imagery: ImagerySource
    ground_truth: GroundTruthSource
    feature: FeatureSpec
    training: TrainingConfig = field(default_factory=TrainingConfig)
    aois: tuple[AOI, ...] = ()
    val_regions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # Normalise: callers may pass bare bbox tuples. `__setattr__`
        # is needed because the dataclass is frozen.
        if self.aois and not all(isinstance(a, AOI) for a in self.aois):
            object.__setattr__(self, "aois", _coerce_aois(self.aois))

    @property
    def cache_namespace(self) -> str:
        """Short namespace combining all three pluggable axes; used as
        a prefix in the on-disk cache so different recipes don't
        clobber each other."""
        return f"{self.imagery.name}__{self.ground_truth.name}__{self.feature.name}"

    @property
    def imagery_cache_namespace(self) -> str:
        """Namespace for the imagery cache. Indexed by imagery only —
        the same (imagery, bbox, max_side, dst_crs) result is reusable
        across recipes that differ only in their GT/feature axis."""
        return self.imagery.name

    @property
    def polygon_cache_namespace(self) -> str:
        """Namespace for the polygon cache. Indexed by (GT source, feature)
        only — polygons depend on the Overpass / vector-tile query and the
        feature definition, but not on which imagery is paired with them.
        Lets sentinel2_parks and hotosm_parks share the same polygon cache.
        """
        return f"{self.ground_truth.name}__{self.feature.name}"

    def _is_val(self, aoi: AOI) -> bool:
        """An AOI is val iff its region is in val_regions OR holdout=True."""
        return aoi.holdout or (bool(self.val_regions) and aoi.region in self.val_regions)

    @property
    def effective_train_aois(self) -> tuple[AOI, ...]:
        """Train AOIs after applying the train/val split.

        Priority:
          1. Any AOI with holdout=True OR region in val_regions -> val
          2. >=2 AOIs and no explicit val signal -> last AOI as val (smoke recipes)
          3. otherwise -> all aois are train
        """
        has_explicit_val = any(self._is_val(a) for a in self.aois)
        if has_explicit_val:
            return tuple(a for a in self.aois if not self._is_val(a))
        if len(self.aois) >= 2:
            return self.aois[:-1]
        return self.aois

    @property
    def effective_val_aois(self) -> tuple[AOI, ...]:
        has_explicit_val = any(self._is_val(a) for a in self.aois)
        if has_explicit_val:
            return tuple(a for a in self.aois if self._is_val(a))
        if len(self.aois) >= 2:
            return (self.aois[-1],)
        return ()

    @property
    def regions(self) -> tuple[str, ...]:
        """Unique, sorted region tags across all AOIs."""
        return tuple(sorted({a.region for a in self.aois if a.region}))

    @property
    def run_dir(self) -> Path:
        """`{training.output_dir}/{name}` — where checkpoints / threshold
        sweeps / predictions for this recipe land."""
        return Path(self.training.output_dir) / self.name

    @classmethod
    def from_spec(
        cls,
        *,
        name: str,
        imagery: str,
        imagery_kwargs: dict | None = None,
        ground_truth: str,
        ground_truth_kwargs: dict | None = None,
        feature: str,
        feature_kwargs: dict | None = None,
        training: TrainingConfig | None = None,
        aois: Iterable = (),
        val_regions: Iterable[str] = (),
    ) -> "Recipe":
        """Convenience builder for recipes defined by short strings.

        Lets `recipes/*.py` files declare a Recipe without having to
        import every concrete class directly.

        `aois` accepts either `AOI(...)` objects or bare bbox 4-tuples
        (the latter get region="" — fine for smoke recipes, not fine
        for production training where region matters).
        """
        return cls(
            name=name,
            imagery=get_imagery_source(imagery, **(imagery_kwargs or {})),
            ground_truth=get_gt_source(ground_truth, **(ground_truth_kwargs or {})),
            feature=get_feature(feature, **(feature_kwargs or {})),
            training=training or TrainingConfig(),
            aois=_coerce_aois(aois),
            val_regions=tuple(val_regions),
        )


def load_recipe(path: str | Path) -> Recipe:
    """Load a recipe by executing a python file and returning its
    `recipe` attribute. Keeps recipes expressive (python > YAML) at
    the cost of trusting the file."""
    p = Path(path).resolve()
    if not p.is_file():
        raise FileNotFoundError(p)
    namespace: dict = {"__file__": str(p)}
    code = compile(p.read_text(encoding="utf-8"), str(p), "exec")
    exec(code, namespace)
    if "recipe" not in namespace:
        raise ValueError(f"{p} did not define a top-level `recipe` variable")
    obj = namespace["recipe"]
    if not isinstance(obj, Recipe):
        raise TypeError(f"{p} `recipe` is not a Recipe (got {type(obj).__name__})")
    return obj
