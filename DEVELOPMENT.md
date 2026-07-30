# CollectorVision Catalog development

This guide is for contributors building and publishing CollectorVision Catalog.
End-user installation and usage belong in the [main README](README.md).

## Current implementation status

The source adapters, deterministic FP16/JSONL artifact builder, exact
predecessor deltas, historical Scryfall replay, and immutable version staging
are implemented.

The family-scoped feed, release audits, and catalog-first Pages layout are
implemented. The remaining Catalog v2 cutover work is:

1. Assign catalog-local versions in the updater and skip unchanged catalogs.
2. Restore scheduled publication after the updater uses the active contract.
3. Adapt the existing CollectorVision Python and browser consumers to the
   catalog-local contract.

The builder now emits compact primary `id` plus optional `face_index` records
with line-aligned metadata. The published historical beta checkpoints have been
rebuilt with this active contract rather than converted from earlier artifacts.

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
  checkpoints, release audits, feed routes, and public paths.
- [`config/catalogs.json`](config/catalogs.json) lists source catalogs.
- [`config/source-quality-overrides.json`](config/source-quality-overrides.json)
  contains reviewed source-image exclusions.

Pages deployment is assembled by
[`.github/workflows/deploy-pages.yml`](.github/workflows/deploy-pages.yml).
The earlier beta producer was removed so that it cannot overwrite the active
feed while catalog-local update publication is being integrated.

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
