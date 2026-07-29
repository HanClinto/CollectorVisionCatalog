# Catalog v2 versioning and paths

Catalog v2 versions belong to individual catalogs, not to the repository as a
whole. Scryfall MTG can remain at version 6 while TCGplayer Digimon advances
from version 12 to 15. A catalog receives a new version only when its client
data changes; a source check that finds no changes updates the feed's
`checked_at` timestamp without creating artifacts.

Catalog v2 is pre-release. This is its active contract; discarded prototypes do
not define compatibility requirements. Catalog v1 is unaffected.

## Public paths

Every catalog has an explicit, unique, lowercase `public_name`. Public URLs use
the catalog first, followed by the singular `version` parameter:

```text
catalog-v2/scryfall-mtg/version/10/manifest.json
catalog-v2/scryfall-mtg/version/10/base/embeddings.f16.gz
catalog-v2/scryfall-mtg/version/10/base/identifiers.jsonl.gz
catalog-v2/scryfall-mtg/version/10/base/metadata.jsonl.gz
catalog-v2/scryfall-mtg/version/10/delta-from-9/embeddings.f16.gz
catalog-v2/scryfall-mtg/version/10/delta-from-9/identifiers.jsonl.gz
catalog-v2/scryfall-mtg/version/10/delta-from-9/metadata.jsonl.gz
```

The repeated `version/10/delta-from-9` is intentional. It makes the exact-base
constraint understandable when a URL is copied, logged, cached, or inspected
without the feed. Dates do not control storage paths; source and publication
timestamps belong in manifests and the feed.

## Version sequence

Versions are non-negative integers scoped to one catalog:

1. Version 0 is the initial checkpoint and contains `base/` only.
2. A changed catalog advances by exactly one version.
3. Ordinary versions contain `delta-from-N/`, where `N` is the immediate
   predecessor.
4. Routine checkpoints contain both `base/` and `delta-from-N/`.
5. Hard checkpoints contain `base/` only and intentionally force every client
   to refresh.

With the default checkpoint interval of 10:

| Version | Published routes |
| ---: | --- |
| 0 | `base/` |
| 1–9 | `delta-from-0/` through `delta-from-8/` |
| 10 | `base/` and `delta-from-9/` |
| 11–19 | predecessor deltas |
| 20 | `base/` and `delta-from-19/` |

The interval is configurable per catalog with `checkpoint_interval`. A routine
checkpoint preserves the cheapest path for a client on the immediately
preceding version while giving new or stale clients a compact starting point.
A hard checkpoint is used for an intentional reset or compaction boundary;
omitting its predecessor delta is what forces the refresh. A different embedding
model starts a distinct catalog identity rather than continuing this version
sequence.

## Manifest arrangement

The manifest groups artifacts by installation route:

```json
{
  "catalog_key": "milo1/scryfall/mtg",
  "public_name": "scryfall-mtg",
  "version": 10,
  "previous_version": 9,
  "source_revision": {
    "type": "scryfall",
    "name": "default_cards",
    "updated_at": "2026-07-28T09:09:18.622Z",
    "uri": "https://data.scryfall.io/default-cards/default-cards-20260728090918.jsonl.gz",
    "identity": "e2ef41e3-5778-4bc2-af3f-78eca4dd9c23"
  },
  "embedding_model": "milo-1.0.0",
  "rows": 110656,
  "dim": 128,
  "dtype": "float16",
  "base": {
    "embeddings": {
      "path": "base/embeddings.f16.gz",
      "size": 25887836,
      "sha256": "..."
    },
    "identifiers": {
      "path": "base/identifiers.jsonl.gz",
      "size": 5710019,
      "sha256": "..."
    },
    "metadata": {
      "path": "base/metadata.jsonl.gz",
      "size": 5177483,
      "sha256": "..."
    }
  },
  "delta": {
    "from_version": 9,
    "operations": 62,
    "metadata_operations": 25,
    "assets": {
      "embeddings": {
        "path": "delta-from-9/embeddings.f16.gz",
        "size": 14665,
        "sha256": "..."
      },
      "identifiers": {
        "path": "delta-from-9/identifiers.jsonl.gz",
        "size": 7119,
        "sha256": "..."
      },
      "metadata": {
        "path": "delta-from-9/metadata.jsonl.gz",
        "size": 1206,
        "sha256": "..."
      }
    }
  }
}
```

`base` and `delta` are independently optional:

| Version kind | `base` | `delta` |
| --- | --- | --- |
| Initial version 0 | object | `null` |
| Incremental version | `null` | object |
| Routine checkpoint | object | object |
| Hard checkpoint | object | `null` |

The top-level row count and dimensions describe the catalog after installing
that version by either available route. `previous_version` records catalog
history even for a hard checkpoint. A delta's `from_version` must equal
`previous_version` and must advance exactly one version.

## Feed routing

The moving feed presents alternatives rather than one mandatory chain:

- the newest usable checkpoint base;
- the bridge delta into that checkpoint when one exists;
- each subsequent predecessor delta;
- freshness timestamps independent of artifact publication.

For current version 12 with a routine checkpoint at 10:

- a new client installs base 10, then deltas 11 and 12;
- a client on version 9 installs deltas 10, 11, and 12;
- a stale client without a supported route installs base 10, then deltas 11
  and 12.

After a hard checkpoint at 17, no route from 16 is advertised. Every client
must install base 17.

Immutable historical versions remain addressable, but the feed only needs to
describe supported paths to the current version.

The feed groups the checkpoint base separately from an ordered predecessor-delta
list. The list may begin with the bridge into the checkpoint, so it is not
defined as a chain that starts from `base.version`:

```json
{
  "checked_at": "2026-07-28T12:00:00Z",
  "catalogs": {
    "milo1/scryfall/mtg": {
      "public_name": "scryfall-mtg",
      "current_version": 12,
      "source_updated_at": "2026-07-28T09:09:18.622Z",
      "base": {
        "version": 10,
        "manifest": {"url": ".../version/10/manifest.json", "size": 1900, "sha256": "..."},
        "assets": {}
      },
      "deltas": [
        {"from_version": 9, "to_version": 10, "manifest": {"url": ".../version/10/manifest.json", "size": 1900, "sha256": "..."}, "assets": {}},
        {"from_version": 10, "to_version": 11, "manifest": {"url": ".../version/11/manifest.json", "size": 1300, "sha256": "..."}, "assets": {}},
        {"from_version": 11, "to_version": 12, "manifest": {"url": ".../version/12/manifest.json", "size": 1300, "sha256": "..."}, "assets": {}}
      ]
    }
  }
}
```

Each delta must be contiguous with the next delta. The first delta may start one
version before `base.version`; clients select the suffix whose `from_version`
matches their installed version. Clients with no matching suffix install
`base`, then select the suffix starting at `base.version`. A hard checkpoint
discards all earlier deltas, so its list begins after the checkpoint.
