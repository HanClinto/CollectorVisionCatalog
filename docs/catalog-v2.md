# CollectorVision Catalog v2

## Goals

Catalog v2 is designed to:

1. Build automatically without retaining a permanent image cache.
2. Download and embed only new or changed source images.
3. Give new clients a complete snapshot and current clients a small,
   one-release delta.
4. Keep recognition data small while allowing optional metadata.
5. Represent peer source identifiers and card faces without implying authority
   or hierarchy among identifier systems.
6. Support many games without placing every artifact in one directory.
7. Preserve Catalog v1 while clients migrate.

## Publication model

Catalog v2 is published from `HanClinto/CollectorVisionCatalog` as immutable
GitHub Releases. A release tag identifies one atomic catalog generation:

```text
catalog-v2-beta.<N>-YYYY-MM-DD
```

Every beta release contains `catalog-index-v2.json`. During beta, clients use an
explicit reviewed tag, such as `catalog-v2-beta.1-2026-07-24`, rather than a
moving `latest` URL. Beta releases are prereleases and do not become stable
`latest`:

```text
https://github.com/HanClinto/CollectorVisionCatalog/releases/download/<tag>/catalog-index-v2.json
```

The beta number increases monotonically. The index maps a stable key such as
`milo1/scryfall/mtg` to that catalog's
manifest and repeats its immutable descriptor: `game`, `source`, `profile`,
`description`, `result_identifier`, and `recommended`. Clients discover
profiles from descriptors, never by parsing catalog keys. Release assets are
flat because GitHub Release assets do not have directories; filenames use a
collision-free catalog slug.

The date suffix is the UTC date of the index's maximum `source_updated_at`, not
the build date. Each index entry and manifest contains the same immutable source
revision (`type`, `name`, normalized `updated_at`, provenance URI, and identity).
The index repeats the maximum timestamp at top level for release tooling.

Hugging Face remains the Catalog v1 host during migration. Catalog v2 does not
replace or mutate v1 manifests.

## Artifact layers

Each catalog manifest references three independent layers.

### Minimal recognition layer

The required download contains:

- little-endian, row-major FP16 embeddings;
- aligned recognition records;
- the embedding model identifier and dimensions.

A recognition record has this logical shape:

```json
{
  "key": "scryfall:00000000-0000-0000-0000-000000000000:face:1",
  "identifiers": {
    "scryfall_card": "00000000-0000-0000-0000-000000000000",
    "scryfall_oracle": "11111111-1111-1111-1111-111111111111",
    "tcgplayer_product": "12345"
  },
  "face_index": 1
}
```

`key` identifies an embedding row and is stable across releases. All external
IDs are peers in `identifiers`; names are explicit, such as `scryfall_card`,
`scryfall_oracle`, `tcgplayer_product`, `tcgplayer_etched_product`,
`tcgplayer_category`, and `tcgplayer_group`. The catalog descriptor chooses the
single `result_identifier` returned by compatibility search APIs, and every row
must contain it. `face_index` is omitted for front faces and defaults to `0`; `1`
identifies the back face. Genuine per-face names belong in optional metadata,
not recognition records.

The matrix is deterministic gzip-compressed raw little-endian, row-major FP16,
not NPZ. Raw FP16 is a shared substrate for NumPy and browsers; NPZ is
ZIP/NumPy-specific and is poor for web streaming. Browsers can keep it packed
and convert values during dot products. Gzip remains a whole-asset download:
assets are immutable and checksummed, and clients cache and replace a profile
atomically. Sharding or range access should be added only after measured need.

The FP16 matrix and recognition JSONL are the required client layer. Metadata
is optional. Builder state is separate and clients must not download it.

### Profiles

Profiles are independent physical snapshots, not filters over one downloaded
catalog. `cards` and `artworks` suit small/mobile clients and stream overlays.
`printings` and `all-languages` suit marketplaces or workflows needing more
edition/language candidates. The recommended Scryfall MTG `printings` profile
uses `default_cards`; optional seed-required profiles use `oracle_cards`,
`unique_artwork`, and `all_cards`. TCGplayer catalogs are `printings` profiles.
Catalog coverage and model discrimination are separate: a larger profile does
not promise accurate language or edition distinction.

The compact Scryfall profiles are configured but disabled until each is
separately seeded. The initial beta may contain only the recommended Scryfall
and TCGplayer `printings` catalogs.

### Optional metadata layer

Metadata is a separate gzip JSONL asset keyed by recognition `key`. Clients
that only need recognition never download it. Common fields include:

- name;
- set ID, code, and name;
- collector number;
- rarity;
- language;
- available finishes.

### Normalized physical attributes

`finishes` is the canonical metadata field for available physical finishes,
following Scryfall's terminology. Values are source-normalized strings such as
`nonfoil`, `foil`, `etched`, and `glossy`. A finish is distinct from:

- `language`, which identifies the language of a specific printing;
- marketplace condition, which does not change recognition identity;
- visual variants such as first edition, shadowless, borderless, or showcase.

Scryfall supplies authoritative `finishes` for each printing. TCGplayer finish
and language availability belongs to SKU data: a product ID groups SKUs across
printing treatment, language, and condition. TCGCSV explicitly does not publish
SKUs, so TCGplayer catalogs built from TCGCSV must omit `finishes` rather than
infer an incomplete list from product-level extended data. Exact TCGplayer
finish support therefore requires a separately reviewed SKU-capable source.

Source-specific fields may be added compatibly, but prices are not part of the
official catalog.

### Updater state

State is a release asset used by the next build. It contains the stable key,
image URL, and image fingerprint for every row. It contains neither source
images nor a cache of source responses.

The state and prior recognition snapshot are sufficient to reuse unchanged
embeddings. Metadata fingerprints are intentionally independent from image
fingerprints: a corrected card name should update metadata without causing an
image download or inference.

## Full snapshots and deltas

Every release publishes:

- a full recognition snapshot;
- full optional metadata;
- updater state;
- a delta whose `base_version` is the immediately preceding release.

Recognition deltas contain `upsert` and `delete` operations. Upserts refer to
rows in a compact FP16 delta matrix. Metadata has its own upsert/delete delta.
A versioned client must apply a delta only when its installed version exactly equals
`base_version`; otherwise it downloads the full snapshot.

A seed manifest has `base_version: null`, zero recognition and metadata
operations, and empty delta assets. A seed is installed from its full snapshot;
`apply_delta(None, seed)` is invalid and does not pretend the empty delta can
reconstruct that snapshot.

This deliberately avoids permanent delta chains. Weekly users get small
updates, while stale clients have a simple and reliable recovery path.
Every weekly release still carries a full snapshot, so new and stale users
never fetch or assemble a chain of deltas.

## Seed assembly

Scryfall and TCGplayer seeds are independently built into distinct directories,
then validated and atomically assembled under one explicit beta version:

```bash
python scripts/assemble_release.py source-status --output source-status.json
SOURCE_DATE="$(python -c 'import json; print(json.load(open("source-status.json"))["suggested_date_suffix"])')"
VERSION="catalog-v2-beta.1-$SOURCE_DATE"

COLLECTORVISION_PROVIDER=cpu python scripts/seed_scryfall.py \
  --version "$VERSION" \
  --expected-source-revisions source-status.json \
  --cache-root /path/to/ccg_card_id/catalog \
  --legacy-catalog /path/to/milo1-scryfall-mtg-latest.npz \
  --output-dir scryfall-seed \
  --build --max-downloads <reviewed-count>

COLLECTORVISION_PROVIDER=cpu python scripts/seed_tcgplayer.py \
  --version "$VERSION" \
  --expected-source-revisions source-status.json \
  --cache-root /path/to/ccg_card_id/catalog \
  --legacy-dir /path/to/ccg_card_id/catalog/tcgplayer/collectorvision \
  --output-dir tcgplayer-seed \
  --build --max-downloads <reviewed-count>

python scripts/assemble_release.py assemble \
  --input-dir scryfall-seed \
  --input-dir tcgplayer-seed \
  --output-dir release \
  --version "$VERSION"
gh release create "$VERSION" --prerelease --latest=false release/*
```

The flat assembled release contains the combined index; all catalog manifests;
full recognition, metadata, and updater-state assets; empty seed delta assets;
merged quality and seed summaries; and deterministic `SHA256SUMS`. Each input
summary is retained with the catalog keys contributed by that input.
Assembly verifies the beta suffix against the combined index timestamp.

After `update_catalogs.py`, the weekly producer validates the full snapshot and
its exact-one-step deltas and refreshes checksums before upload:

```bash
python scripts/assemble_release.py validate \
  --release-dir release --version "$VERSION" --write-checksums
```

## Build algorithm

For each configured source/game/model tuple:

1. Verify the captured upstream revision and stream current source metadata.
2. Normalize each usable source image into a stable recognition record.
3. Compare each row's image fingerprint with the previous release state.
4. Reuse the previous embedding when both key and image fingerprint match.
5. Download changed images into bounded temporary batches.
6. Embed each batch and immediately discard its images.
7. Sort by stable key and write deterministic artifacts.
8. Build and verify the one-step delta against the full snapshot.
9. Publish all catalogs as one atomic GitHub Release.

If no valid previous release exists, the job requires an explicit seed input
instead of silently attempting a full hosted-run rebuild.

### Source image quality

Source records may be valid while their images are unsuitable for recognition,
including placeholders, text overlays, watermarks, and non-camera-true
reference art. Versioned quality rules can approve, quarantine, or reject rows
by source, category, group, product, face, and metadata name pattern. Rules can
exclude reviewed exceptions from a broader match. Quarantined and rejected rows
are removed before embedding and listed in `quality-report.json`; these
decisions do not add fields to minimal recognition records.

Reviewed rules quarantine TCGplayer groups whose card images contain a large
added `Not Tournament Legal` annotation. Magic World Championship biography,
decklist, and blank inserts are preserved because their images do not contain
the annotation. Longer-term reviewed camera observations and replacement
references are tracked in
[issue #1](https://github.com/HanClinto/CollectorVisionCatalog/issues/1).

## Sources

### Scryfall

The MTG adapter uses Scryfall bulk data. Each available face is a separate
recognition row. Scryfall card UUIDs, Oracle IDs, and available TCGplayer IDs are peer
identifiers.

The selected bulk-data entry supplies the source type, bulk identity,
`jsonl_download_uri`, and update timestamp. The bulk JSONL is streamed as a
build input and is not included in releases.

Image fingerprints include Scryfall's image revision timestamp. A revised
source image therefore invalidates its embedding. Metadata-only changes do not
invalidate embeddings. Automated updates have a changed-row safety budget and
abort before downloads when an unexpectedly broad upstream refresh requires a
reviewed local rebuild.

Local seed builds use the persistent `ccg_card_id` Scryfall cache. Cache paths
are resolved directly from the card ID and face rather than by scanning the
full cache. URL timestamps are compared with local file modification times.
Stale and missing files are refreshed with bounded concurrency and written
atomically before inference.

### TCGCSV

TCGplayer-backed games use the free TCGCSV product feed. The updater checks
`last-updated.txt` immediately before and after fetching categories, groups, and
products and aborts if it changes, so all categories share one global revision.
It sends a descriptive user agent and respects TCGCSV's request guidance.
TCGplayer product, category, and group IDs are peer identifiers.

TCGCSV payloads are streamed as build inputs and are not release assets.
Together, recorded source identities and timestamps make baseline seeds and
later refreshes auditable and reproducible by upstream snapshot.

The seed migrates the eight existing Milo catalogs without refreshing their
legacy front images. Existing product embeddings are reused; only new products
and additional faces use freshly downloaded images. TCGCSV `modifiedOn` values
are part of image fingerprints so revisions after the v2 baseline trigger
incremental embedding updates. Prices are not downloaded.

## Compatibility and rollout

1. Publish and validate seeded v2 snapshots without changing CollectorVision.
2. Add an opt-in v2 client to CollectorVision Python and JavaScript.
3. Make new embedding models v2-only.
4. Keep Milo Catalog v1 manifests available for a documented transition
   period.
5. Retire v1 only after usage and issue reports show that supported clients
   have migrated.

Catalog schemas are versioned independently from embedding models. Clients
must reject unsupported major schema versions rather than guessing.
