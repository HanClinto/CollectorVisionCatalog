# CollectorVision Catalog v2

## Goals

Catalog v2 is designed to:

1. Build automatically without retaining a permanent image cache.
2. Download and embed only new or changed source images.
3. Give new clients a recent complete checkpoint and current clients exact
   predecessor deltas.
4. Keep recognition data small while allowing optional metadata.
5. Represent peer source identifiers and card faces without implying authority
   or hierarchy among identifier systems.
6. Support many games without placing every artifact in one directory.
7. Preserve Catalog v1 while clients migrate.

## Publication model

Catalog v2 is published from `HanClinto/CollectorVisionCatalog`. Immutable
versions advance independently for each catalog only when that catalog changes:

```text
catalog-v2/<public-name>/version/<N>/
```

The moving feed is published at:

```text
https://hanclinto.github.io/CollectorVisionCatalog/catalog-v2/catalog-feed-v2.json
```

`checked_at` advances even when no catalog changes. Each catalog entry records
its own source freshness and supported routes to its current version. Dates are
manifest metadata, not storage identity.

The complete catalog-local numbering rules, routine and hard checkpoints,
public paths, feed routing, and manifest structure are defined in
[Catalog v2 versioning and paths](versioning.md). Discarded prototypes do not
receive aliases or fallback parsers.

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
  "id": "00000000-0000-0000-0000-000000000000",
  "identifiers": {
    "scryfall_oracle": "11111111-1111-1111-1111-111111111111",
    "tcgplayer_product": "12345"
  },
  "face_index": 1
}
```

`id` is the catalog's primary result ID. The catalog descriptor's
`result_identifier` names its source namespace—for example, `scryfall_card` or
`tcgplayer_product`—without repeating that name and value in every row. Other
source IDs remain explicit peers in `identifiers`, such as `scryfall_oracle`,
`tcgplayer_product`, `tcgplayer_etched_product`, `tcgplayer_category`, and
`tcgplayer_group`.

The selected result namespace must be the provider's primary namespace:
Scryfall catalogs use `scryfall_card`, and TCGplayer catalogs use
`tcgplayer_product`. To look up a row by that namespace, clients compare the
query with `id`; `identifiers` contains only additional namespaces.

`face_index` is omitted for front faces and defaults to `0`; `1` identifies the
back face. A client that needs a globally unique map key can derive
`<provider>:<id>` for the front face and `<provider>:<id>:face:<N>` for
additional faces. The derived key is not stored in public rows. Genuine
per-face names belong in optional metadata, not recognition records.

The matrix is deterministic gzip-compressed raw little-endian, row-major FP16,
not NPZ. Raw FP16 is a shared substrate for NumPy and browsers; NPZ is
ZIP/NumPy-specific and is poor for web streaming. Browsers can keep it packed
and convert values during dot products. Gzip remains a whole-asset download:
assets are immutable and checksummed, and clients cache and replace a profile
atomically. Sharding or range access should be added only after measured need.

The FP16 embeddings and identifiers are the required client layer. Metadata
is optional. Builder state is separate and clients must not download it.

### Catalog coverage

Scryfall provides one default MTG recognition catalog using `default_cards`.
It retains distinct artworks and printings. Promo, token, art-card, finish, and
similar policies are lookup-time filters, not separate physical catalogs.
Catalog coverage and model discrimination remain separate: including a
printing does not promise accurate language or edition distinction.

### Optional metadata layer

Metadata is a separate gzip JSONL asset aligned one-to-one with recognition
rows. Each line is the row's metadata object, or JSON `null` when that row has
no metadata. It does not repeat row identity. Clients that only need
recognition never download it. Common fields include:

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

State is a private builder asset used by the next build. It is line-aligned
with recognition rows and contains the image URL and image fingerprint without
repeating row identity. It contains neither source images nor a cache of source
responses.

The state and prior recognition snapshot are sufficient to reuse unchanged
embeddings. Metadata fingerprints are intentionally independent from image
fingerprints: a corrected card name should update metadata without causing an
image download or inference.

## Full snapshots and deltas

Version 0 contains a complete base. Changed catalogs normally publish only an
exact-predecessor delta. At the configured checkpoint interval—10 versions by
default—a routine checkpoint publishes both a complete base and the predecessor
delta. Existing clients continue incrementally while new clients start from the
new base.

A hard checkpoint publishes a base without a delta and deliberately forces a
full refresh. See [Catalog v2 versioning and paths](versioning.md) for the exact
rules and manifest shapes.

Delta operations cannot rely on row alignment, so they target a row with `id`
and an optional nonzero `face_index`. Recognition upserts carry changed
identifiers and embeddings; metadata upserts carry changed metadata. Deletes
use the same compact target. The provider remains catalog-level manifest data.
Recognition and metadata operations are independent and idempotent, so a
removed row may have a delete in both layers.

### Reconstructing historical Scryfall snapshots

Timestamped `source_revision.uri` values in published manifests can be replayed
without changing the normal catalog builder. Use a config with only the Scryfall
catalog enabled, then provide an archived `.json`, `.json.gz`, `.jsonl`, or
`.jsonl.gz` file or URL:

```bash
python scripts/update_catalogs.py \
  --config /path/to/scryfall-only.json \
  --previous-dir /path/to/empty-directory \
  --output-dir historical-base \
  --version 0 \
  --allow-full-rebuild \
  --scryfall-bulk-uri https://data.scryfall.io/default-cards/default-cards-20260720123456.jsonl.gz \
  --scryfall-bulk-updated-at 2026-07-20T12:34:56Z \
  --cache-root /path/to/image-cache

python scripts/update_catalogs.py \
  --config /path/to/scryfall-only.json \
  --previous-dir historical-base \
  --output-dir historical-update \
  --version 1 \
  --scryfall-bulk-uri /path/to/default-cards-20260721123456.jsonl.gz \
  --scryfall-bulk-updated-at 2026-07-21T12:34:56Z \
  --cache-root /path/to/image-cache
```

The second build uses the reconstructed first snapshot as its exact base and
produces normal identifier, embedding, and metadata deltas. `--scryfall-bulk-format`
can explicitly select `json` or `jsonl` when a filename has no recognizable
extension. `--scryfall-bulk-identity` can record an archive-specific identifier;
otherwise the normalized file or download URI is retained as the identity.

## Build algorithm

For each configured source/game/model tuple:

1. Verify the captured upstream revision and stream current source metadata.
2. Normalize each usable source image into a stable recognition record.
3. Compare each row's image fingerprint with the previous release state.
4. Reuse the previous embedding when both derived row identity and image
   fingerprint match.
5. Download changed images into bounded temporary batches.
6. Embed each batch and immediately discard its images.
7. Sort by derived row identity and write deterministic artifacts.
8. Build and verify the exact-predecessor delta against the full snapshot.
9. Publish each changed catalog version and update the moving feed.

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
recognition row. The Scryfall card UUID is the primary `id`; Oracle IDs and
available TCGplayer IDs are peer identifiers.

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

The TCGplayer product ID is the primary `id`; category, group, and other
available source IDs are peer identifiers. TCGCSV payloads are streamed as
build inputs and are not release assets.
Together, recorded source identities and timestamps make baseline seeds and
later refreshes auditable and reproducible by upstream snapshot.

The seed migrates the eight existing Milo catalogs without refreshing their
legacy front images. Existing product embeddings are reused; only new products
and additional faces use freshly downloaded images. TCGCSV `modifiedOn` values
are part of image fingerprints so revisions after the v2 baseline trigger
incremental embedding updates. Prices are not downloaded.

## Compatibility and rollout

CollectorVision Python and browser consumers have not yet been updated to this
active contract. They must gain integer catalog versions, alternative
checkpoint routes, and catalog-first URLs before the beta is testable end to
end.

Catalog v1 remains supported independently. Catalog v2 has no compatibility
requirement before its stable release, so discarded prototype code will be
replaced rather than retained as aliases or fallback parsers.
