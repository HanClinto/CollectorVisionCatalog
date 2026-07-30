#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import runpy
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from collectorvision_catalog import (
    ValidationError,
    assemble_catalog_release,
    load_catalog_version_manifest,
    max_source_updated_at,
    validate_catalog_release,
)
from collectorvision_catalog.release import (
    assemble_seed_release,
    validate_release,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble or validate Catalog v2 releases")
    subparsers = parser.add_subparsers(dest="command", required=True)

    assemble = subparsers.add_parser("assemble", help="assemble independent seed builds")
    assemble.add_argument("--input-dir", type=Path, action="append", required=True)
    assemble.add_argument("--output-dir", type=Path, required=True)
    assemble.add_argument("--version", required=True)

    audit = subparsers.add_parser(
        "assemble-audit",
        help="assemble public assets and one release-wide audit from private receipts",
    )
    audit.add_argument(
        "--publication",
        action="append",
        nargs=3,
        required=True,
        metavar=("CATALOG_KEY", "VERSION_ROOT", "RECEIPT"),
    )
    audit.add_argument(
        "--input",
        action="append",
        nargs=2,
        required=True,
        metavar=("NAME", "PATH"),
    )
    audit.add_argument("--output-dir", type=Path, required=True)
    audit.add_argument("--tag", required=True)
    audit.add_argument("--published-at", required=True)
    audit.add_argument("--repository", required=True)
    audit.add_argument("--commit", required=True)

    validate = subparsers.add_parser("validate", help="validate an existing release")
    validate.add_argument("--release-dir", type=Path, required=True)
    validate.add_argument("--version")
    validate_audit = subparsers.add_parser(
        "validate-audit",
        help="validate a public release against its audit",
    )
    validate_audit.add_argument("--release-dir", type=Path, required=True)
    status = subparsers.add_parser(
        "source-status",
        help="fetch enabled source revisions without downloading catalog rows",
    )
    status.add_argument("--config", type=Path, default=Path("config/catalogs.json"))
    status.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "assemble":
        assemble_seed_release(args.input_dir, args.output_dir, args.version)
    elif args.command == "assemble-audit":
        publications = {}
        for catalog_key, version_root, receipt_path in args.publication:
            if catalog_key in publications:
                raise ValidationError(f"duplicate publication {catalog_key!r}")
            publications[catalog_key] = (
                Path(version_root),
                load_catalog_version_manifest(receipt_path),
            )
        inputs = {}
        for name, path in args.input:
            if name in inputs:
                raise ValidationError(f"duplicate audit input {name!r}")
            inputs[name] = Path(path)
        assemble_catalog_release(
            publications,
            args.output_dir,
            tag=args.tag,
            published_at=args.published_at,
            repository=args.repository,
            commit=args.commit,
            inputs=inputs,
        )
    elif args.command == "source-status":
        updater = runpy.run_path(str(Path(__file__).with_name("update_catalogs.py")))
        configs = [config for config in updater["load_config"](args.config) if config.enabled]
        if not configs:
            raise ValidationError("config does not enable any catalogs")
        revisions = {}
        scryfall_by_type = {}
        tcgcsv_revision = None
        for config in configs:
            source_type = config.source.get("type")
            if source_type == "scryfall":
                bulk_type = config.source.get("bulk_type")
                revision = scryfall_by_type.get(bulk_type)
                if revision is None:
                    revision = updater["fetch_scryfall_revision"](config.source)
                    scryfall_by_type[bulk_type] = revision
            elif source_type == "tcgcsv":
                if tcgcsv_revision is None:
                    tcgcsv_revision = updater["fetch_tcgcsv_revision"]()
                revision = tcgcsv_revision
            else:
                raise ValidationError(f"unsupported source type {source_type!r}")
            revisions[config.key] = revision
        source_updated_at = max_source_updated_at(revisions.values())
        payload = {
            "checked_at": datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z"),
            "source_updated_at": source_updated_at,
            "suggested_date_suffix": source_updated_at[:10],
            "catalogs": {key: revision.to_dict() for key, revision in sorted(revisions.items())},
        }
        rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        if args.output is not None:
            args.output.write_text(rendered, encoding="utf-8")
        print(rendered, end="")
    elif args.command == "validate":
        validate_release(args.release_dir, expected_version=args.version)
    else:
        validate_catalog_release(args.release_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
