#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import runpy
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
from numpy.typing import NDArray

from collectorvision_catalog import (
    RecognitionRow,
    SourceRevision,
    ValidationError,
    build_catalog,
    manifest_filename_for_catalog,
    validate_artifacts,
    write_catalog_index,
)
from collectorvision_catalog.quality import apply_quality_rules, load_quality_rules

_UPDATER = runpy.run_path(str(Path(__file__).with_name("update_catalogs.py")))
ScryfallImageCache = _UPDATER["ScryfallImageCache"]
create_embedder = _UPDATER["create_embedder"]
fetch_scryfall_snapshot = _UPDATER["fetch_scryfall_snapshot"]
load_config = _UPDATER["load_config"]
scryfall_source_override = _UPDATER["_scryfall_source_override"]


@dataclass(frozen=True)
class SeedPlan:
    rows: tuple[RecognitionRow, ...]
    seed_embeddings: Mapping[str, NDArray[np.float32]]
    inference_rows: tuple[RecognitionRow, ...]
    download_rows: tuple[RecognitionRow, ...]
    cache_current: int
    cache_stale: int
    cache_missing: int

    @property
    def downloads_required(self) -> int:
        return self.cache_stale + self.cache_missing

    def summary(self) -> dict[str, int]:
        return {
            "current_rows": len(self.rows),
            "legacy_embeddings_reused": len(self.seed_embeddings),
            "embeddings_to_compute": len(self.inference_rows),
            "cache_current": self.cache_current,
            "cache_stale": self.cache_stale,
            "cache_missing": self.cache_missing,
            "downloads_required": self.downloads_required,
        }


def create_seed_plan(
    rows: Sequence[RecognitionRow],
    image_cache: Any,
    legacy_embeddings: Mapping[str, NDArray[np.float32]] | None = None,
) -> SeedPlan:
    legacy_embeddings = legacy_embeddings or {}
    current_keys = {row.key for row in rows}
    reusable = {
        key: embedding for key, embedding in legacy_embeddings.items() if key in current_keys
    }
    inference_rows = tuple(row for row in rows if row.key not in reusable)
    download_rows = tuple({row.image_url: row for row in inference_rows}.values())
    cache_current = 0
    cache_stale = 0
    cache_missing = 0
    for row in download_rows:
        path = image_cache.path_for_row(row)
        if not path.is_file():
            cache_missing += 1
        elif image_cache.is_current(row):
            cache_current += 1
        else:
            cache_stale += 1
    return SeedPlan(
        rows=tuple(rows),
        seed_embeddings=reusable,
        inference_rows=inference_rows,
        download_rows=download_rows,
        cache_current=cache_current,
        cache_stale=cache_stale,
        cache_missing=cache_missing,
    )


def refresh_seed_cache(
    plan: SeedPlan,
    image_cache: Any,
    *,
    workers: int,
) -> None:
    pending = [row for row in plan.download_rows if not image_cache.is_current(row)]
    if not pending:
        return
    if workers <= 0:
        raise ValidationError("refresh workers must be positive")

    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(image_cache, row.image_url): row
            for row in pending
        }
        try:
            for future in as_completed(futures):
                image = future.result()
                image.close()
                completed += 1
                if completed % 1000 == 0 or completed == len(pending):
                    print(f"Refreshed {completed:,}/{len(pending):,} Scryfall images", flush=True)
        except BaseException:
            for future in futures:
                future.cancel()
            raise


def build_seed(
    *,
    config_path: Path,
    quality_overrides_path: Path,
    cache_root: Path,
    legacy_catalog: Path | None,
    output_dir: Path,
    version: str,
    batch_size: int,
    max_downloads: int,
    refresh_workers: int,
    build: bool,
    expected_revision: SourceRevision | None = None,
    source_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    configs = [
        config
        for config in load_config(config_path)
        if config.enabled and config.source.get("type") == "scryfall"
    ]
    if len(configs) != 1:
        raise ValidationError("seed requires exactly one enabled Scryfall catalog")
    config = configs[0]
    snapshot = fetch_scryfall_snapshot({**config.source, **(source_override or {})})
    if expected_revision is not None and snapshot.revision != expected_revision:
        raise ValidationError("Scryfall source revision does not match expected revision")
    rows = list(snapshot.rows)
    quality_result = apply_quality_rules(
        rows,
        source_type="scryfall",
        rules=load_quality_rules(quality_overrides_path),
    )
    rows = list(quality_result.rows)
    image_cache = ScryfallImageCache(cache_root, rows)
    legacy_embeddings = (
        {} if legacy_catalog is None else load_legacy_embeddings(legacy_catalog)
    )
    plan = create_seed_plan(rows, image_cache, legacy_embeddings)
    summary: dict[str, Any] = {
        "catalog_key": config.key,
        "version": version,
        "source_revision": snapshot.revision.to_dict(),
        **plan.summary(),
        "quality_excluded_rows": len(quality_result.findings),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not build:
        return summary
    if plan.downloads_required > max_downloads:
        raise ValidationError(
            f"seed requires {plan.downloads_required:,} downloads, exceeding the explicit "
            f"limit of {max_downloads:,}; rerun with --max-downloads after reviewing preflight"
        )

    refresh_seed_cache(plan, image_cache, workers=refresh_workers)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = build_catalog(
        plan.rows,
        embedder=create_embedder(config.embedding_model, batch_size),
        output_dir=output_dir,
        catalog_key=config.key,
        version=version,
        embedding_model=config.embedding_model,
        source_revision=snapshot.revision,
        descriptor=config.descriptor,
        seed_embeddings=plan.seed_embeddings,
        image_loader=image_cache,
        batch_size=batch_size,
    )
    manifest_path = output_dir / manifest_filename_for_catalog(config.key)
    validate_artifacts(manifest_path, asset_dir=output_dir)
    index = write_catalog_index(
        output_dir / "catalog-index-v2.json",
        version,
        {config.key: manifest_path},
    )
    summary["source_updated_at"] = index.source_updated_at
    summary["output_rows"] = result.manifest.rows
    (output_dir / "seed-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "quality-report.json").write_text(
        json.dumps(
            {
                "version": version,
                "catalogs": {
                    config.key: {
                        **quality_result.report(),
                        "source_revision": snapshot.revision.to_dict(),
                    }
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def load_legacy_embeddings(path: Path) -> dict[str, NDArray[np.float32]]:
    try:
        catalog = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValidationError(f"cannot load legacy Scryfall catalog {path}: {error}") from error
    with catalog:
        if "embeddings" not in catalog or "card_ids" not in catalog:
            raise ValidationError("legacy Scryfall catalog must contain embeddings and card_ids")
        embeddings = np.asarray(catalog["embeddings"], dtype=np.float32)
        raw_card_ids = np.asarray(catalog["card_ids"])
        if embeddings.ndim != 2 or len(embeddings) != len(raw_card_ids):
            raise ValidationError("legacy Scryfall embeddings and card_ids are not aligned")
        if raw_card_ids.ndim == 1:
            card_ids = [str(value) for value in raw_card_ids]
        elif raw_card_ids.ndim == 2 and raw_card_ids.shape[1] == 16:
            if raw_card_ids.dtype != np.uint8:
                raise ValidationError("packed legacy Scryfall card_ids must use uint8")
            card_ids = [str(UUID(bytes=bytes(row))) for row in raw_card_ids]
        else:
            raise ValidationError(
                "legacy Scryfall card_ids must be strings or packed 16-byte UUID rows"
            )
        if "source" in catalog and str(catalog["source"].item()) != "scryfall":
            raise ValidationError("legacy catalog source must be 'scryfall'")
        if "embedder_spec" in catalog:
            try:
                embedder_spec = json.loads(str(catalog["embedder_spec"].item()))
            except (ValueError, json.JSONDecodeError) as error:
                raise ValidationError("legacy Scryfall embedder_spec is invalid") from error
            if embedder_spec.get("algo_key") != "milo1":
                raise ValidationError("legacy Scryfall catalog must use Milo 1 embeddings")

        result: dict[str, NDArray[np.float32]] = {}
        for index, card_id in enumerate(card_ids):
            face_index = 1 if card_id.endswith("_back") else 0
            source_id = card_id.removesuffix("_back")
            try:
                canonical_id = str(UUID(source_id))
            except ValueError as error:
                raise ValidationError(
                    f"legacy Scryfall card_ids[{index}] is not a UUID"
                ) from error
            if source_id.lower() != canonical_id:
                raise ValidationError(
                    f"legacy Scryfall card_ids[{index}] is not canonical"
                )
            key = f"scryfall:{canonical_id}:face:{face_index}"
            if key in result:
                raise ValidationError(f"duplicate legacy Scryfall row key {key!r}")
            result[key] = embeddings[index]
    return result


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
        "--legacy-catalog",
        type=Path,
        default=None,
        help="Optional Catalog v1 Scryfall NPZ whose matching Milo embeddings are reused",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("release"))
    parser.add_argument("--version", required=True)
    parser.add_argument("--expected-source-revisions", type=Path)
    parser.add_argument(
        "--scryfall-bulk-uri",
        help="Archived Scryfall .json[.gz] or .jsonl[.gz] file path or URL",
    )
    parser.add_argument("--scryfall-bulk-updated-at")
    parser.add_argument("--scryfall-bulk-identity")
    parser.add_argument("--scryfall-bulk-format", choices=("json", "jsonl"))
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
    source_override = scryfall_source_override(args)
    expected_revision = None
    if args.expected_source_revisions is not None:
        payload = json.loads(args.expected_source_revisions.read_text(encoding="utf-8"))
        configs = [
            config
            for config in load_config(args.config)
            if config.enabled and config.source.get("type") == "scryfall"
        ]
        if len(configs) != 1:
            raise ValidationError("seed requires exactly one enabled Scryfall catalog")
        raw_catalogs = payload.get("catalogs")
        if not isinstance(raw_catalogs, dict) or configs[0].key not in raw_catalogs:
            raise ValidationError("expected source revisions are missing Scryfall catalog")
        expected_revision = SourceRevision.from_dict(raw_catalogs[configs[0].key])
    build_seed(
        config_path=args.config,
        quality_overrides_path=args.quality_overrides,
        cache_root=args.cache_root,
        legacy_catalog=args.legacy_catalog,
        output_dir=args.output_dir,
        version=args.version,
        batch_size=args.batch_size,
        max_downloads=args.max_downloads,
        refresh_workers=args.refresh_workers,
        build=args.build,
        expected_revision=expected_revision,
        source_override=source_override,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
