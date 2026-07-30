from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .artifacts import CatalogError, load_manifest, validate_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cvcatalog")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Print a manifest summary as JSON")
    inspect_parser.add_argument("manifest", type=Path)

    validate_parser = subparsers.add_parser("validate", help="Validate a manifest and its assets")
    validate_parser.add_argument("manifest", type=Path)
    validate_parser.add_argument("--asset-dir", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            manifest = load_manifest(args.manifest)
            print(json.dumps(manifest.to_dict(), indent=2))
            return 0
        if args.command == "validate":
            build = validate_artifacts(args.manifest, asset_dir=args.asset_dir)
            print(
                f"Validated {build.manifest.catalog_key}@{build.manifest.version} "
                f"({build.manifest.rows} rows, dim={build.manifest.dim})"
            )
            return 0
    except CatalogError as error:
        parser.exit(status=1, message=f"cvcatalog: {error}\n")
    raise AssertionError(f"unhandled command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
