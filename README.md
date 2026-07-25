# CollectorVision Catalog

Automated, incremental recognition catalogs for
[CollectorVision](https://github.com/HanClinto/CollectorVision).

Catalog v2 separates the data needed for recognition from optional card
metadata:

- **Recognition:** raw FP16 embeddings plus aligned rows containing a stable
  per-face key and peer, explicitly named source identifiers.
- **Metadata:** names, sets, languages, finishes, and other display fields.
- **Updates:** a full snapshot for new installations plus a one-release delta
  for existing installations.

Images are build inputs, not release assets. A weekly build compares upstream
records with the previous release, downloads and embeds only new or changed
images, reuses unchanged embeddings, and then publishes immutable gzip assets
to a GitHub Release.

Scryfall and TCGCSV source data is streamed during builds, not redistributed.
Every manifest, index entry, summary, and quality report records the upstream
source identity, provenance URL, and normalized UTC update timestamp.

Versioned rules in `config/source-quality-overrides.json` quarantine annotated,
placeholder, or otherwise unsuitable source images before embedding. Builds
emit `quality-report.json` with the affected row keys and reasons; quality data
stays outside compact recognition records.

## Status

Catalog v2 is a beta discovered through an explicit release tag. CollectorVision
Catalog v1 remains
available from Hugging Face and is not changed by this repository.

Catalog descriptors in the release index and manifests expose independent
physical profiles. Compact Scryfall `cards` and `artworks` profiles are
configured but disabled until they receive separate seeds. The initial beta may
therefore contain only the recommended Scryfall and TCGplayer `printings`
catalogs. Profile availability does not imply that an embedding model can
reliably distinguish every edition or language.

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

Capture the enabled source identities first. The latest source timestamp's UTC
date, not the build date, is the beta suffix:

```bash
python -m pip install -e .
python -m pip install onnxruntime \
  "collectorvision[hf] @ git+https://github.com/HanClinto/CollectorVision.git@9d45a37ebfe40f22ece70507015645de134dc3ec"

python scripts/assemble_release.py source-status --output source-status.json
SOURCE_DATE="$(python -c 'import json; print(json.load(open("source-status.json"))["suggested_date_suffix"])')"
VERSION="catalog-v2-beta.1-$SOURCE_DATE"

COLLECTORVISION_PROVIDER=cpu python scripts/seed_scryfall.py \
  --version "$VERSION" \
  --expected-source-revisions source-status.json \
  --cache-root /path/to/ccg_card_id/catalog \
  --legacy-catalog /path/to/milo1-scryfall-mtg-latest.npz \
  --output-dir scryfall-seed
```

That command is a preflight and does not download images or run inference.
Matching front and `_back` rows reuse the pinned Milo Catalog v1 embeddings.
Review its staleness and cache coverage counts, then repeat it with `--build`
and an explicit `--max-downloads` no lower than the reported requirement. The
seed builder reads the existing sharded `scryfall/images/png/front|back` cache,
refreshes stale files with bounded concurrency, and writes downloads back
atomically.

The eight existing TCGplayer catalogs can be migrated separately while reusing
legacy Milo embeddings for existing product images:

```bash
COLLECTORVISION_PROVIDER=cpu python scripts/seed_tcgplayer.py \
  --version "$VERSION" \
  --expected-source-revisions source-status.json \
  --cache-root /path/to/ccg_card_id/catalog \
  --legacy-dir /path/to/ccg_card_id/catalog/tcgplayer/collectorvision \
  --output-dir tcgplayer-seed
```

This is also a preflight by default. New products and additional faces are
downloaded from the CDN and embedded only after `--build` and a reviewed
`--max-downloads` are supplied. Existing legacy product images remain untouched.
After reviewing both preflights, repeat each command with `--build` and its
reviewed `--max-downloads`, then assemble and publish:

```bash
python scripts/assemble_release.py assemble \
  --input-dir scryfall-seed \
  --input-dir tcgplayer-seed \
  --output-dir release \
  --version "$VERSION"
gh release create "$VERSION" --prerelease --latest=false release/*
```

The assembler validates every manifest and asset before atomically exposing
`release/`. The flat release contains one combined index, each catalog manifest,
full recognition snapshots, optional full metadata, updater state, empty seed
deltas, merged quality and seed reports, and deterministic `SHA256SUMS`.
It also rejects a beta tag whose date differs from the index's maximum upstream
UTC update date. Expected revisions make a source change between status and
fetch abort instead of publishing a misleading tag.

Each later weekly beta also contains a complete snapshot plus a delta from
exactly the preceding beta. Current clients may apply that one step; new or
stale clients download the full snapshot and never assemble delta chains.

See [the Catalog v2 protocol](docs/catalog-v2.md) for the artifact and update
design.

## Local image-quality OCR

On macOS, compile the thin Apple Vision wrapper once and stream image paths
through it. Output is JSONL so audits can retain text evidence without coupling
CI to a macOS-only backend:

```bash
swiftc -O scripts/apple_vision_ocr.swift -o /tmp/collectorvision-apple-ocr
find /path/to/images -type f -name '*.jpg' |
  /tmp/collectorvision-apple-ocr --stdin |
  gzip -1 > apple-vision-ocr.jsonl.gz
```
