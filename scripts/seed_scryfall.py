#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import runpy
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from collectorvision_catalog import (
    RecognitionRow,
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
fetch_scryfall_rows = _UPDATER["fetch_scryfall_rows"]
load_config = _UPDATER["load_config"]


@dataclass(frozen=True)
class SeedPlan:
    rows: tuple[RecognitionRow, ...]
    cache_current: int
    cache_stale: int
    cache_missing: int

    @property
    def downloads_required(self) -> int:
        return self.cache_stale + self.cache_missing

    def summary(self) -> dict[str, int]:
        return {
            "current_rows": len(self.rows),
            "embeddings_to_compute": len(self.rows),
            "cache_current": self.cache_current,
            "cache_stale": self.cache_stale,
            "cache_missing": self.cache_missing,
            "downloads_required": self.downloads_required,
        }


def create_seed_plan(
    rows: Sequence[RecognitionRow],
    image_cache: Any,
) -> SeedPlan:
    cache_current = 0
    cache_stale = 0
    cache_missing = 0
    for row in rows:
        path = image_cache.path_for_row(row)
        if not path.is_file():
            cache_missing += 1
        elif image_cache.is_current(row):
            cache_current += 1
        else:
            cache_stale += 1
    return SeedPlan(
        rows=tuple(rows),
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
    pending = [row for row in plan.rows if not image_cache.is_current(row)]
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
    output_dir: Path,
    version: str,
    batch_size: int,
    max_downloads: int,
    refresh_workers: int,
    build: bool,
) -> dict[str, Any]:
    configs = [
        config
        for config in load_config(config_path)
        if config.enabled and config.source.get("type") == "scryfall"
    ]
    if len(configs) != 1:
        raise ValidationError("seed requires exactly one enabled Scryfall catalog")
    config = configs[0]
    rows = fetch_scryfall_rows(config.source)
    quality_result = apply_quality_rules(
        rows,
        source_type="scryfall",
        rules=load_quality_rules(quality_overrides_path),
    )
    rows = list(quality_result.rows)
    image_cache = ScryfallImageCache(cache_root, rows)
    plan = create_seed_plan(rows, image_cache)
    summary: dict[str, Any] = {
        "catalog_key": config.key,
        "version": version,
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
        image_loader=image_cache,
        batch_size=batch_size,
    )
    manifest_path = output_dir / manifest_filename_for_catalog(config.key)
    validate_artifacts(manifest_path, asset_dir=output_dir)
    write_catalog_index(
        output_dir / "catalog-index-v2.json",
        version,
        {config.key: manifest_path},
    )
    summary["output_rows"] = result.manifest.rows
    (output_dir / "seed-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "quality-report.json").write_text(
        json.dumps(
            {
                "version": version,
                "catalogs": {config.key: quality_result.report()},
            },
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
    parser.add_argument("--output-dir", type=Path, default=Path("release"))
    parser.add_argument("--version", required=True)
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
    build_seed(
        config_path=args.config,
        quality_overrides_path=args.quality_overrides,
        cache_root=args.cache_root,
        output_dir=args.output_dir,
        version=args.version,
        batch_size=args.batch_size,
        max_downloads=args.max_downloads,
        refresh_workers=args.refresh_workers,
        build=args.build,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
