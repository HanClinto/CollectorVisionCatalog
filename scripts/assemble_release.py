#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import runpy
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from collectorvision_catalog import ValidationError, max_source_updated_at
from collectorvision_catalog.release import (
    assemble_seed_release,
    validate_release,
    write_checksums,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble or validate Catalog v2 releases")
    subparsers = parser.add_subparsers(dest="command", required=True)

    assemble = subparsers.add_parser("assemble", help="assemble independent seed builds")
    assemble.add_argument("--input-dir", type=Path, action="append", required=True)
    assemble.add_argument("--output-dir", type=Path, required=True)
    assemble.add_argument("--version", required=True)

    validate = subparsers.add_parser("validate", help="validate an existing release")
    validate.add_argument("--release-dir", type=Path, required=True)
    validate.add_argument("--version")
    validate.add_argument("--write-checksums", action="store_true")
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
    elif args.write_checksums:
        validate_release(args.release_dir, expected_version=args.version)
        write_checksums(args.release_dir)
        validate_release(args.release_dir, expected_version=args.version)
    else:
        validate_release(args.release_dir, expected_version=args.version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
