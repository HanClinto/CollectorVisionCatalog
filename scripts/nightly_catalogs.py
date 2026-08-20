#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

from collectorvision_catalog import (
    CatalogBuild,
    CatalogFeed,
    SourceRevision,
    ValidationError,
    advance_catalog_feed,
    assemble_catalog_release,
    load_catalog_build,
    load_catalog_feed,
    manifest_filename_for_catalog,
    plan_catalog_version,
    publish_catalog_version,
    validate_catalog_release,
    write_catalog_feed,
)


def _load_updater() -> ModuleType:
    path = Path(__file__).with_name("update_catalogs.py")
    spec = importlib.util.spec_from_file_location("collectorvision_catalog_updater", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load catalog updater from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _feed_entries(feed: CatalogFeed) -> dict[str, Any]:
    return {
        f"{family_name}/{local_key}": entry
        for family_name, family in feed.families.items()
        for local_key, entry in family.catalogs.items()
    }


def catalog_versions(feed: CatalogFeed) -> dict[str, str]:
    return {
        key: str(entry.current_version + 1)
        for key, entry in _feed_entries(feed).items()
    }


def finalize_nightly(
    *,
    feed: CatalogFeed,
    summary: Mapping[str, Any],
    previous_dir: Path,
    build_dir: Path,
    public_dir: Path,
    release_dir: Path,
    next_state_dir: Path,
    output_feed: Path,
    tag: str,
    published_at: str,
    repository: str,
    commit: str,
    inputs: Mapping[str, Path],
    checked_at: str,
    checkpoint_interval: int = 10,
    retained_deltas: int = 30,
) -> dict[str, Any]:
    changed_keys = [
        item["catalog_key"] for item in summary["catalogs"] if item.get("changed") is True
    ]
    if not changed_keys:
        return {"changed": False, "tag": None, "catalogs": {}}
    if retained_deltas < checkpoint_interval:
        raise ValidationError("retained_deltas must be at least checkpoint_interval")
    entries = _feed_entries(feed)

    publications = {}
    next_builds: dict[str, tuple[CatalogBuild, Path]] = {}
    for catalog_key in entries:
        manifest_name = manifest_filename_for_catalog(catalog_key)
        candidate_dir = build_dir if catalog_key in changed_keys else previous_dir
        manifest_path = candidate_dir / manifest_name
        if not manifest_path.is_file():
            raise ValidationError(f"builder state is missing catalog {catalog_key!r}")
        next_builds[catalog_key] = (
            load_catalog_build(manifest_path, asset_dir=candidate_dir),
            candidate_dir,
        )

    for catalog_key in changed_keys:
        entry = entries.get(catalog_key)
        if entry is None:
            raise ValidationError(f"changed catalog {catalog_key!r} is not present in the feed")
        build, source_dir = next_builds[catalog_key]
        previous_path = previous_dir / manifest_filename_for_catalog(catalog_key)
        previous = load_catalog_build(previous_path, asset_dir=previous_dir)
        plan = plan_catalog_version(
            entry.current_version,
            checkpoint_interval=checkpoint_interval,
        )
        if build.manifest.version != str(plan.version):
            raise ValidationError(f"builder version for {catalog_key!r} does not match its plan")
        receipt, version_dir = publish_catalog_version(
            build,
            source_dir,
            public_dir,
            entry.public_name,
            plan,
            previous_build=previous,
        )
        publications[catalog_key] = (version_dir, receipt)

    assemble_catalog_release(
        publications,
        release_dir,
        tag=tag,
        published_at=published_at,
        repository=repository,
        commit=commit,
        inputs=inputs,
    )
    validate_catalog_release(release_dir)
    advanced = advance_catalog_feed(
        feed,
        publications,
        checked_at=checked_at,
        retained_deltas=retained_deltas,
    )
    write_catalog_feed(output_feed, advanced)
    _write_builder_state(next_builds, next_state_dir)
    return {
        "changed": True,
        "tag": tag,
        "catalogs": {
            key: {
                "version": manifest.version,
                "rows": manifest.rows,
            }
            for key, (_, manifest) in sorted(publications.items())
        },
    }


def _write_builder_state(
    builds: Mapping[str, tuple[CatalogBuild, Path]],
    destination: Path,
) -> None:
    if destination.exists():
        raise ValidationError(f"next builder state already exists: {destination}")
    destination.mkdir(parents=True)
    for catalog_key, (build, source_dir) in sorted(builds.items()):
        manifest_name = manifest_filename_for_catalog(catalog_key)
        shutil.copy2(source_dir / manifest_name, destination / manifest_name)
        for asset in build.manifest.assets.values():
            shutil.copy2(source_dir / asset.filename, destination / asset.filename)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and stage changed catalog-local versions")
    parser.add_argument("--config", type=Path, default=Path("config/catalogs.json"))
    parser.add_argument(
        "--quality-overrides",
        type=Path,
        default=Path("config/source-quality-overrides.json"),
    )
    parser.add_argument("--feed", type=Path, default=Path("catalog-feed-v2.json"))
    parser.add_argument("--previous-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-feed", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--next-state-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--source-status", type=Path)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--published-at")
    parser.add_argument("--repository", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--retained-deltas", type=int, default=30)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    updater = _load_updater()
    feed = load_catalog_feed(args.feed)
    expected_revisions = _load_source_revisions(args.source_status)
    build_dir = args.work_dir / "build"
    summary = updater.build_enabled_catalogs(
        config_path=args.config,
        quality_overrides_path=args.quality_overrides,
        previous_dir=args.previous_dir,
        output_dir=build_dir,
        version=catalog_versions(feed),
        cache_root=args.cache_root,
        batch_size=args.batch_size,
        expected_source_revisions=expected_revisions,
        skip_unchanged=True,
    )
    published_at = args.published_at or _now()
    inputs = {
        "catalog_config": args.config,
        "quality_overrides": args.quality_overrides,
        "update_summary": build_dir / "update-summary.json",
        "quality_report": build_dir / "quality-report.json",
    }
    if args.source_status is not None:
        inputs["source_status"] = args.source_status
    result = finalize_nightly(
        feed=feed,
        summary=summary,
        previous_dir=args.previous_dir,
        build_dir=build_dir,
        public_dir=args.work_dir / "public",
        release_dir=args.release_dir,
        next_state_dir=args.next_state_dir,
        output_feed=args.output_feed,
        tag=args.tag,
        published_at=published_at,
        repository=args.repository,
        commit=args.commit,
        inputs=inputs,
        checked_at=published_at,
        checkpoint_interval=args.checkpoint_interval,
        retained_deltas=args.retained_deltas,
    )
    result["source_updated_at"] = summary["source_updated_at"]
    result_path = args.work_dir / "nightly-summary.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _load_source_revisions(path: Path | None) -> dict[str, SourceRevision]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    catalogs = payload.get("catalogs")
    if not isinstance(catalogs, dict):
        raise ValidationError("source status catalogs must be an object")
    return {str(key): SourceRevision.from_dict(value) for key, value in catalogs.items()}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
