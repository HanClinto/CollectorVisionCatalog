#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import runpy
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from collectorvision_catalog import (
    RecognitionRow,
    SourceRevision,
    ValidationError,
    build_catalog,
    catalog_row_key,
    manifest_filename_for_catalog,
    validate_artifacts,
    write_catalog_index,
)
from collectorvision_catalog.quality import apply_quality_rules, load_quality_rules

_UPDATER = runpy.run_path(str(Path(__file__).with_name("update_catalogs.py")))
TCGplayerImageCache = _UPDATER["TCGplayerImageCache"]
TCGplayerImageUnavailable = _UPDATER["TCGplayerImageUnavailable"]
create_embedder = _UPDATER["create_embedder"]
fetch_tcgcsv_snapshots = _UPDATER["fetch_tcgcsv_snapshots"]
load_config = _UPDATER["load_config"]

LEGACY_CATALOG_KEYS = {
    "milo1/tcgplayer/magic-the-gathering": "tcgplayer-mtg",
    "milo1/tcgplayer/yugioh": "tcgplayer-yugioh",
    "milo1/tcgplayer/pokemon": "tcgplayer-pokemon",
    "milo1/tcgplayer/flesh-and-blood": "tcgplayer-fab",
    "milo1/tcgplayer/digimon-card-game": "tcgplayer-digimon",
    "milo1/tcgplayer/one-piece": "tcgplayer-onepiece",
    "milo1/tcgplayer/lorcana": "tcgplayer-lorcana",
    "milo1/tcgplayer/star-wars-unlimited": "tcgplayer-swu",
}


@dataclass(frozen=True)
class TCGplayerSeedPlan:
    catalog_key: str
    rows: tuple[RecognitionRow, ...]
    seed_embeddings: Mapping[str, NDArray[np.float32]]
    inference_rows: tuple[RecognitionRow, ...]
    download_rows: tuple[RecognitionRow, ...]

    def summary(self) -> dict[str, Any]:
        return {
            "catalog_key": self.catalog_key,
            "current_rows": len(self.rows),
            "legacy_embeddings_reused": len(self.seed_embeddings),
            "downloads_required": len(self.download_rows),
            "embeddings_to_compute": len(self.inference_rows),
        }


def load_legacy_embeddings(
    legacy_dir: Path,
    legacy_catalog_key: str,
) -> dict[str, NDArray[np.float32]]:
    manifest_path = legacy_dir / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = payload.get(legacy_catalog_key)
    if not isinstance(entry, dict) or not isinstance(entry.get("latest"), str):
        raise ValidationError(f"legacy manifest does not contain {legacy_catalog_key!r}")
    catalog_path = legacy_dir / entry["latest"]
    with np.load(catalog_path, allow_pickle=False) as catalog:
        card_ids = catalog["card_ids"]
        embeddings = catalog["embeddings"]
        if card_ids.ndim != 1 or embeddings.ndim != 2 or len(card_ids) != len(embeddings):
            raise ValidationError(f"invalid legacy catalog arrays in {catalog_path}")
        if embeddings.shape[1] != 128:
            raise ValidationError(
                f"legacy catalog {catalog_path} has unexpected dimension {embeddings.shape[1]}"
            )
        result: dict[str, NDArray[np.float32]] = {}
        for card_id, embedding in zip(card_ids, embeddings, strict=True):
            key = catalog_row_key("tcgplayer", card_id)
            if key in result:
                raise ValidationError(f"duplicate legacy card ID {card_id!r} in {catalog_path}")
            result[key] = embedding.astype(np.float32, copy=True)
    return result


def create_seed_plan(
    catalog_key: str,
    rows: Sequence[RecognitionRow],
    legacy_embeddings: Mapping[str, NDArray[np.float32]],
    image_cache: Any,
) -> TCGplayerSeedPlan:
    current_keys = {row.key for row in rows}
    reusable: dict[str, NDArray[np.float32]] = {}
    inference_rows: list[RecognitionRow] = []
    for row in rows:
        embedding = legacy_embeddings.get(row.key)
        if embedding is not None and row.face_index == 0:
            reusable[row.key] = embedding
        else:
            inference_rows.append(row)
    unknown_legacy_rows = len(set(legacy_embeddings).difference(current_keys))
    if unknown_legacy_rows:
        print(
            f"{catalog_key}: {unknown_legacy_rows:,} legacy rows are absent from current TCGCSV",
            flush=True,
        )
    return TCGplayerSeedPlan(
        catalog_key=catalog_key,
        rows=tuple(rows),
        seed_embeddings=reusable,
        inference_rows=tuple(inference_rows),
        download_rows=tuple(row for row in inference_rows if not image_cache.is_cached(row)),
    )


def refresh_inference_images(
    rows: Iterable[RecognitionRow],
    image_cache: Any,
    *,
    workers: int,
    catalog_key: str,
) -> set[str]:
    pending = tuple(rows)
    if not pending:
        return set()
    if workers <= 0:
        raise ValidationError("refresh workers must be positive")
    unavailable_keys: set[str] = set()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(image_cache, row.image_url): row for row in pending}
        try:
            for completed, future in enumerate(as_completed(futures), start=1):
                row = futures[future]
                try:
                    image = future.result()
                except TCGplayerImageUnavailable as error:
                    unavailable_keys.add(row.key)
                    print(f"{catalog_key}: skipping {row.key}: {error}", flush=True)
                    continue
                image.close()
                if completed % 1000 == 0 or completed == len(pending):
                    print(
                        f"{catalog_key}: refreshed {completed:,}/{len(pending):,} images",
                        flush=True,
                    )
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    return unavailable_keys


def build_seed(
    *,
    config_path: Path,
    quality_overrides_path: Path,
    cache_root: Path,
    legacy_dir: Path | None,
    output_dir: Path,
    version: str,
    batch_size: int,
    max_downloads: int,
    refresh_workers: int,
    build: bool,
    catalog_keys: Sequence[str] | None = None,
    expected_revision: SourceRevision | None = None,
) -> dict[str, Any]:
    configs_by_key = {
        config.key: config
        for config in load_config(config_path)
        if config.source.get("type") == "tcgcsv"
    }
    selected_keys = tuple(catalog_keys or LEGACY_CATALOG_KEYS)
    if not selected_keys:
        raise ValidationError("at least one TCGplayer seed catalog must be selected")
    if len(selected_keys) != len(set(selected_keys)):
        raise ValidationError("TCGplayer seed catalog selections must be unique")
    missing_configs = sorted(set(selected_keys).difference(configs_by_key))
    if missing_configs:
        raise ValidationError(f"missing TCGplayer seed configs: {missing_configs}")
    legacy_keys = set(selected_keys).intersection(LEGACY_CATALOG_KEYS)
    if legacy_keys and legacy_dir is None:
        raise ValidationError("--legacy-dir is required when seeding legacy catalogs")

    plans: list[TCGplayerSeedPlan] = []
    model_ids: set[str] = set()
    quality_reports: dict[str, Any] = {}
    quality_rules = load_quality_rules(quality_overrides_path)
    snapshots = fetch_tcgcsv_snapshots(
        {catalog_key: configs_by_key[catalog_key].source for catalog_key in selected_keys},
        expected_revision=expected_revision,
    )
    for catalog_key in selected_keys:
        config = configs_by_key[catalog_key]
        snapshot = snapshots[catalog_key]
        rows = list(snapshot.rows)
        quality_result = apply_quality_rules(
            rows,
            source_type="tcgcsv",
            rules=quality_rules,
        )
        rows = list(quality_result.rows)
        quality_reports[catalog_key] = {
            **quality_result.report(),
            "source_revision": snapshot.revision.to_dict(),
        }
        legacy_key = LEGACY_CATALOG_KEYS.get(catalog_key)
        legacy_embeddings = (
            load_legacy_embeddings(legacy_dir, legacy_key)
            if legacy_key is not None and legacy_dir is not None
            else {}
        )
        image_cache = TCGplayerImageCache(cache_root, rows)
        known_unavailable = {row.key for row in rows if image_cache.is_temporarily_unavailable(row)}
        if known_unavailable:
            print(
                f"{catalog_key}: excluding {len(known_unavailable):,} recently unavailable images",
                flush=True,
            )
        available_rows = [row for row in rows if row.key not in known_unavailable]
        plan = create_seed_plan(catalog_key, available_rows, legacy_embeddings, image_cache)
        plans.append(plan)
        model_ids.add(config.embedding_model)
        print(json.dumps(plan.summary(), indent=2, sort_keys=True), flush=True)

    total_downloads = sum(len(plan.download_rows) for plan in plans)
    summary: dict[str, Any] = {
        "version": version,
        "source_revision": next(iter(snapshots.values())).revision.to_dict(),
        "catalogs": [plan.summary() for plan in plans],
        "downloads_required": total_downloads,
        "embeddings_to_compute": sum(len(plan.inference_rows) for plan in plans),
        "legacy_embeddings_reused": sum(len(plan.seed_embeddings) for plan in plans),
        "current_rows": sum(len(plan.rows) for plan in plans),
        "quality_excluded_rows": sum(
            report["excluded_rows"] for report in quality_reports.values()
        ),
    }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if not build:
        return summary
    if total_downloads > max_downloads:
        raise ValidationError(
            f"seed requires {total_downloads:,} downloads, exceeding the explicit limit of "
            f"{max_downloads:,}; rerun with --max-downloads after reviewing preflight"
        )
    if len(model_ids) != 1:
        raise ValidationError("all TCGplayer seed catalogs must use the same embedding model")

    output_dir.mkdir(parents=True, exist_ok=True)
    embedder = create_embedder(model_ids.pop(), batch_size)
    manifests: dict[str, Path] = {}
    for plan in plans:
        refreshing_cache = TCGplayerImageCache(cache_root, plan.rows)
        unavailable_keys = refresh_inference_images(
            plan.download_rows,
            refreshing_cache,
            workers=refresh_workers,
            catalog_key=plan.catalog_key,
        )
        effective_rows = tuple(row for row in plan.rows if row.key not in unavailable_keys)
        effective_inference_rows = tuple(
            row for row in plan.inference_rows if row.key not in unavailable_keys
        )
        effective_seed_embeddings = {
            key: embedding
            for key, embedding in plan.seed_embeddings.items()
            if key not in unavailable_keys
        }
        image_cache = TCGplayerImageCache(cache_root, effective_rows)
        config = configs_by_key[plan.catalog_key]
        build_result = build_catalog(
            effective_rows,
            embedder=embedder,
            output_dir=output_dir,
            catalog_key=plan.catalog_key,
            version=version,
            embedding_model=config.embedding_model,
            source_revision=snapshots[plan.catalog_key].revision,
            descriptor=config.descriptor,
            seed_embeddings=effective_seed_embeddings,
            image_loader=image_cache,
            batch_size=batch_size,
        )
        manifest_path = output_dir / manifest_filename_for_catalog(plan.catalog_key)
        validate_artifacts(manifest_path, asset_dir=output_dir)
        if build_result.manifest.rows != len(effective_rows):
            raise AssertionError("TCGplayer seed output row count changed during build")
        manifests[plan.catalog_key] = manifest_path
        catalog_summary = next(
            item for item in summary["catalogs"] if item["catalog_key"] == plan.catalog_key
        )
        catalog_summary["unavailable_images"] = len(unavailable_keys)
        catalog_summary["output_rows"] = len(effective_rows)
        catalog_summary["embeddings_computed"] = len(effective_inference_rows)

    index = write_catalog_index(output_dir / "catalog-index-v2.json", version, manifests)
    summary["source_updated_at"] = index.source_updated_at
    summary["unavailable_images"] = sum(
        item.get("unavailable_images", 0) for item in summary["catalogs"]
    )
    summary["output_rows"] = sum(item["output_rows"] for item in summary["catalogs"])
    (output_dir / "seed-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "quality-report.json").write_text(
        json.dumps(
            {"version": version, "catalogs": quality_reports},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/catalogs.json"))
    parser.add_argument(
        "--quality-overrides",
        type=Path,
        default=Path("config/source-quality-overrides.json"),
    )
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument(
        "--legacy-dir",
        type=Path,
        help="Legacy v1 catalog directory; required only for catalogs with reusable embeddings",
    )
    parser.add_argument(
        "--catalog",
        dest="catalog_keys",
        action="append",
        help="Catalog key to seed; repeat to select multiple (defaults to legacy catalogs)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("tcgplayer-release"))
    parser.add_argument("--version", required=True)
    parser.add_argument("--expected-source-revisions", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--refresh-workers", type=int, default=4)
    parser.add_argument(
        "--max-downloads",
        type=int,
        default=1000,
        help="Safety limit; the build aborts before refresh when preflight exceeds it",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Refresh and build after preflight; without this flag only print the plan",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    selected_keys = tuple(args.catalog_keys or LEGACY_CATALOG_KEYS)
    expected_revision = None
    if args.expected_source_revisions is not None:
        payload = json.loads(args.expected_source_revisions.read_text(encoding="utf-8"))
        raw_catalogs = payload.get("catalogs")
        if not isinstance(raw_catalogs, dict):
            raise ValidationError("expected source revisions catalogs must be an object")
        revisions = {
            SourceRevision.from_dict(raw_catalogs[key])
            for key in selected_keys
            if key in raw_catalogs
        }
        if len(revisions) != 1 or not set(selected_keys).issubset(raw_catalogs):
            raise ValidationError(
                "expected source revisions must contain one shared TCGCSV revision"
            )
        expected_revision = next(iter(revisions))
    build_seed(
        config_path=args.config,
        quality_overrides_path=args.quality_overrides,
        cache_root=args.cache_root,
        legacy_dir=args.legacy_dir,
        output_dir=args.output_dir,
        version=args.version,
        batch_size=args.batch_size,
        max_downloads=args.max_downloads,
        refresh_workers=args.refresh_workers,
        build=args.build,
        catalog_keys=selected_keys,
        expected_revision=expected_revision,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
