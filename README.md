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

python scripts/update_catalogs.py \
  --version catalog-v2-YYYY-MM-DD \
  --allow-full-rebuild \
  --image-dir /path/to/scryfall/images
```

The updater uses cached files named with the source ID, including the existing
`<scryfall-id>_back` convention. Missing files are downloaded temporarily.
After reviewing `release/update-summary.json`, publish every file in `release/`
to a release with the same version tag. The scheduled workflow will use that
release as its incremental base.

See [the Catalog v2 protocol](docs/catalog-v2.md) for the artifact and update
design.
