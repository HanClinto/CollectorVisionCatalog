# CollectorVision Catalog

Automated, incremental recognition catalogs for
[CollectorVision](https://github.com/HanClinto/CollectorVision).

Catalog v2 separates the data needed for recognition from optional card
metadata:

- **Recognition:** FP16 embeddings, primary IDs, secondary IDs, and face data.
- **Metadata:** names, sets, languages, finishes, and other display fields.
- **Updates:** a full snapshot for new installations plus a one-release delta
  for existing installations.

Images are build inputs, not release assets. A weekly build compares upstream
records with the previous release, downloads and embeds only new or changed
images, reuses unchanged embeddings, and then publishes immutable gzip assets
to a GitHub Release.

Versioned rules in `config/source-quality-overrides.json` quarantine annotated,
placeholder, or otherwise unsuitable source images before embedding. Builds
emit `quality-report.json` with the affected row keys and reasons; quality data
stays outside compact recognition records.

## Status

Catalog v2 is under active development. CollectorVision Catalog v1 remains
available from Hugging Face and is not changed by this repository.

The first production release will be seeded from the existing local Milo
catalogs. Scheduled builds should only be enabled for a catalog after that seed
release exists.

## Development

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
```

## Seeding the first release

The weekly workflow intentionally does nothing until a v2 seed release exists.
Build the seed on a machine with an ONNX Runtime backend and, optionally, your
existing image cache:

```bash
python -m pip install -e .
python -m pip install onnxruntime \
  "collectorvision[hf] @ git+https://github.com/HanClinto/CollectorVision.git@9d45a37ebfe40f22ece70507015645de134dc3ec"

COLLECTORVISION_PROVIDER=cpu python scripts/seed_scryfall.py \
  --version catalog-v2-YYYY-MM-DD \
  --cache-root /path/to/ccg_card_id/catalog
```

That command is a preflight and does not download images or run inference.
Review its staleness and cache coverage counts, then repeat it with `--build`
and an explicit `--max-downloads` no lower than the reported requirement. The
seed builder reads the existing sharded `scryfall/images/png/front|back` cache,
refreshes stale files with bounded concurrency, and writes downloads back
atomically. After reviewing `release/seed-summary.json`, publish every file in
`release/` to a release with the same version tag. The scheduled workflow will
use that release as its incremental base and abort before downloading when an
upstream refresh exceeds the configured safety limit.

The eight existing TCGplayer catalogs can be migrated separately while reusing
legacy Milo embeddings for existing product images:

```bash
COLLECTORVISION_PROVIDER=cpu python scripts/seed_tcgplayer.py \
  --version catalog-v2-YYYY-MM-DD \
  --cache-root /path/to/ccg_card_id/catalog \
  --legacy-dir /path/to/ccg_card_id/catalog/tcgplayer/collectorvision
```

This is also a preflight by default. New products and additional faces are
downloaded from the CDN and embedded only after `--build` and a reviewed
`--max-downloads` are supplied. Existing legacy product images remain untouched.

See [the Catalog v2 protocol](docs/catalog-v2.md) for the artifact and update
design.
