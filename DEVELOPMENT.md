# CollectorVision Catalog development

This guide is for contributors building and publishing CollectorVision Catalog.
End-user installation and usage belong in the [main README](README.md).

## Current implementation status

The source adapters, deterministic FP16/JSONL artifact builder, exact
predecessor deltas, historical Scryfall replay, and immutable version staging
are implemented.

The current clients and base feed exercise the earlier beta layout. The
remaining Catalog v2 cutover work is:

1. Assign catalog-local versions in the updater and skip unchanged catalogs.
2. Replace the beta feed with catalog-local base and delta routes.
3. Publish catalog-first paths through Pages.
4. Adapt the existing CollectorVision Python and browser consumers to the
   catalog-local contract.

Catalog v2 is unreleased, so discarded prototypes do not require compatibility
aliases or fallback parsers. Catalog v1 must remain unchanged.

## Set up the repository

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
```

## Architecture

- [`docs/catalog-v2.md`](docs/catalog-v2.md) describes artifacts, sources,
  metadata boundaries, and the update algorithm.
- [`docs/versioning.md`](docs/versioning.md) defines catalog-local versions,
  checkpoints, manifests, feed routes, and public paths.
- [`config/catalogs.json`](config/catalogs.json) lists source catalogs.
- [`config/source-quality-overrides.json`](config/source-quality-overrides.json)
  contains reviewed source-image exclusions.

The scheduled producer is
[`.github/workflows/weekly-release.yml`](.github/workflows/weekly-release.yml).
It runs Mondays at 10:17 UTC. Pages deployment is assembled by
[`.github/workflows/deploy-pages.yml`](.github/workflows/deploy-pages.yml).

## Historical Scryfall replay

Published Scryfall source revisions can be replayed from `.json`, `.json.gz`,
`.jsonl`, or `.jsonl.gz` archives:

```bash
python scripts/update_catalogs.py \
  --config /path/to/scryfall-only.json \
  --previous-dir /path/to/empty-directory \
  --output-dir historical-base \
  --version 0 \
  --allow-full-rebuild \
  --scryfall-bulk-uri /path/to/default-cards-old.json.gz \
  --scryfall-bulk-updated-at 2026-07-20T12:34:56Z \
  --cache-root /path/to/image-cache

python scripts/update_catalogs.py \
  --config /path/to/scryfall-only.json \
  --previous-dir historical-base \
  --output-dir historical-update \
  --version 1 \
  --scryfall-bulk-uri /path/to/default-cards-new.json.gz \
  --scryfall-bulk-updated-at 2026-07-21T12:34:56Z \
  --cache-root /path/to/image-cache
```

Use `--scryfall-bulk-format json|jsonl` when the filename does not identify the
format. `--scryfall-bulk-identity` records an archive-specific identity.

## Local image-quality OCR

On macOS, compile the Apple Vision wrapper and stream image paths through it:

```bash
swiftc -O scripts/apple_vision_ocr.swift -o /tmp/collectorvision-apple-ocr
find /path/to/images -type f -name '*.jpg' |
  /tmp/collectorvision-apple-ocr --stdin |
  gzip -1 > apple-vision-ocr.jsonl.gz
```

Output is JSONL so audits can retain text evidence without coupling CI to a
macOS-only backend.
