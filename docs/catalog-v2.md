# CollectorVision Catalog v2

## Goals

Catalog v2 is designed to:

1. Build automatically without retaining a permanent image cache.
2. Download and embed only new or changed source images.
3. Give new clients a complete snapshot and current clients a small,
   one-release delta.
4. Keep recognition data small while allowing optional metadata.
5. represent primary IDs, secondary IDs, and card faces consistently.
6. Support many games without placing every artifact in one directory.
7. Preserve Catalog v1 while clients migrate.

## Publication model

Catalog v2 is published from `HanClinto/CollectorVisionCatalog` as immutable
GitHub Releases. A release tag identifies one atomic catalog generation:

```text
catalog-v2-YYYY-MM-DD
```

Every release contains `catalog-index-v2.json`. Clients discover the current
generation through:

```text
https://github.com/HanClinto/CollectorVisionCatalog/releases/latest/download/catalog-index-v2.json
```

The index maps a stable key such as `milo1/scryfall/mtg` to that catalog's
manifest. Release assets are flat because GitHub Release assets do not have
directories; filenames therefore use a collision-free catalog slug.

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
  "key": "scryfall:00000000-0000-0000-0000-000000000000:1",
  "primary_id": {
    "namespace": "scryfall",
    "value": "00000000-0000-0000-0000-000000000000"
  },
  "secondary_ids": {
    "scryfall_oracle": "11111111-1111-1111-1111-111111111111",
    "tcgplayer_product": "12345"
  },
  "face_index": 1
}
```

`key` identifies an embedding row and is stable across releases. `primary_id`
identifies the source card or product. Secondary IDs are namespaced and
optional. `face_index` is omitted for front faces and defaults to `0`; `1`
identifies the back face. Genuine per-face names belong in optional metadata,
not recognition records.

The browser can keep the embedding matrix packed as FP16 and convert values
during dot products. For the current MTG catalog, gzip-compressed FP16 is about
25.7 MB, compared with 56.2 MB for the current compressed FP32 NPZ.

### Optional metadata layer

Metadata is a separate gzip JSONL asset keyed by recognition `key`. Clients
that only need recognition never download it. Common fields include:

- name;
- set ID, code, and name;
- collector number;
- rarity;
- language;
- available finishes or foilings.

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
A client must apply a delta only when its installed version exactly equals
`base_version`; otherwise it downloads the full snapshot.

This deliberately avoids permanent delta chains. Weekly users get small
updates, while stale clients have a simple and reliable recovery path.

## Build algorithm

For each configured source/game/model tuple:

1. Download current upstream metadata.
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
by source, category, group, product, and face. Quarantined and rejected rows are
removed before embedding and listed in `quality-report.json`; these decisions
do not add fields to minimal recognition records.

The initial rule quarantines all 303 image rows from the 302 products in
TCGplayer Magic International Edition group 1527 because its references contain
a large `Not Tournament Legal` annotation. Longer-term reviewed camera
observations and replacement references are tracked in
[issue #1](https://github.com/HanClinto/CollectorVisionCatalog/issues/1).

## Sources

### Scryfall

The MTG adapter uses Scryfall bulk data. Each available face is a separate
recognition row. Scryfall card UUIDs are primary IDs; Oracle and TCGplayer IDs
are secondary IDs.

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
`last-updated.txt`, fetches categories, groups, and products at most once per
day, sends a descriptive user agent, and respects TCGCSV's request guidance.
TCGplayer product IDs are primary IDs; category and group IDs are secondary
IDs.

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
