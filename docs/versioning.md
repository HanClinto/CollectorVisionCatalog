# Catalog v2 versioning and publication

Catalog versions belong to individual catalogs. A catalog advances only when
its client data changes; source checks with no changes update the feed's
`checked_at` timestamp without creating artifacts.

Catalog v2 is pre-release. This is its active contract; discarded prototypes do
not define compatibility requirements. Catalog v1 is unaffected.

## Embedding families

Catalogs are nested under an immutable embedding family:

```json
{
  "families": {
    "milo1": {
      "embedding": {
        "model": "collectorvision@...:milo-1.0.0@sha256:...",
        "dimensions": 128,
        "dtype": "float16",
        "byte_order": "little",
        "layout": "row-major"
      },
      "catalogs": {}
    }
  }
}
```

Every catalog in `milo1` uses that exact model and binary contract. Any model,
preprocessing, dimension, dtype, or layout change creates a new family such as
`milo2`; it never mutates `milo1`.

## Public paths

Every catalog has a unique lowercase `public_name`. Assets use catalog-local
integer versions:

```text
catalog-v2/scryfall-mtg/version/10/base/embeddings.f16.gz
catalog-v2/scryfall-mtg/version/10/base/identifiers.jsonl.gz
catalog-v2/scryfall-mtg/version/10/base/metadata.jsonl.gz
catalog-v2/scryfall-mtg/version/10/delta-from-9/embeddings.f16.gz
catalog-v2/scryfall-mtg/version/10/delta-from-9/identifiers.jsonl.gz
catalog-v2/scryfall-mtg/version/10/delta-from-9/metadata.jsonl.gz
```

The repeated `version/10/delta-from-9` makes the exact predecessor constraint
clear when a URL is copied, cached, or logged without the feed.

## Version sequence

1. Version 0 is the initial checkpoint and contains a base.
2. A changed catalog advances by exactly one version.
3. Ordinary versions contain an exact-predecessor update.
4. Routine checkpoints contain both a base and the predecessor update.
5. Hard checkpoints contain only a base and force a full refresh.

A different embedding contract starts a different family rather than
continuing the existing version sequence.

## Client feed

`catalog-feed-v2.json` is the only document normal clients need to check. It is
a complete discovery and routing projection generated from immutable release
audits.

```json
{
  "checked_at": "2026-07-29T22:40:13Z",
  "families": {
    "milo1": {
      "embedding": {
        "model": "collectorvision@...:milo-1.0.0@sha256:...",
        "dimensions": 128,
        "dtype": "float16",
        "byte_order": "little",
        "layout": "row-major"
      },
      "catalogs": {
        "scryfall/mtg": {
          "public_name": "scryfall-mtg",
          "descriptor": {
            "game": "magic-the-gathering",
            "source": "scryfall",
            "profile": "printings",
            "description": "English-first paper printings.",
            "result_identifier": "scryfall_card",
            "recommended": true
          },
          "current_version": 11,
          "rows": 110681,
          "source_updated_at": "2026-07-29T09:10:00Z",
          "base": {
            "version": 10,
            "rows": 110656,
            "source_updated_at": "2026-07-28T09:09:18Z",
            "recognition": {"assets": {}},
            "metadata": {"assets": {}}
          },
          "updates": {
            "11": {
              "from_version": 10,
              "to_version": 11,
              "rows": {"added": 25, "updated": 37, "deleted": 0},
              "source_updated_at": "2026-07-29T09:10:00Z",
              "recognition": {"rows": 62, "assets": {}},
              "metadata": {"rows": 25, "assets": {}}
            }
          }
        }
      }
    }
  }
}
```

Real asset objects contain `url`, compressed byte `size`, and `sha256`.
Recognition groups the mandatory embeddings and identifiers; metadata records
are an optional client download. Base `rows` states the aligned snapshot size
once. Update `rows` classifies each affected catalog row globally. Layer `rows`
states the number of operations in that layer.

The feed advertises the newest usable checkpoint, its optional bridge update,
and each subsequent predecessor update. Clients with no matching installed
version download the base and then apply updates after `base.version`.

## Release audits

GitHub releases contain one `audit.json`, not one public manifest per catalog
version and not a separate `SHA256SUMS`.

The audit is an immutable provenance receipt containing:

- release tag, publication time, repository, and generator commit;
- hashes of catalog configuration and quality overrides;
- the exact embedding contract for each included family;
- complete source revisions and replay URLs;
- catalog versions, predecessors, resulting rows, and row changes;
- every release asset's filename, size, and SHA-256.

Only catalogs published by that release appear in its audit. Normal clients do
not download audits. The Catalog Explorer may load them on demand for history.
GitHub also records each release asset digest independently.

Private builder manifests and state remain build inputs for validation,
embedding reuse, and future updates. They are not public client artifacts.
