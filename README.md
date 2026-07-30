# CollectorVision Catalog

Ready-to-search card recognition data for
[CollectorVision](https://github.com/HanClinto/CollectorVision).

Choose a game, load its catalog, and identify cards from image embeddings.
Catalog v2 keeps recognition downloads small, offers optional card metadata,
and updates changed cards without repeatedly downloading the entire catalog.

> [!NOTE]
> Catalog v2 is in beta. Browse supported games and recent changes in the
> [Catalog Explorer](https://hanclinto.github.io/CollectorVisionCatalog/).

## Use it

Install CollectorVision from GitHub during the beta:

```bash
python -m pip install \
  "collectorvision @ git+https://github.com/HanClinto/CollectorVision.git"
```

<details>
<summary><strong>Python example</strong></summary>

```python
from PIL import Image
import collector_vision as cv

catalog = cv.CatalogV2("mtg", include_metadata=True)

with Image.open("card.jpg") as image:
    embedding = catalog.embedder.embed(image.convert("RGB"))

match = catalog.search_records(embedding, top_k=1)[0]
print(match["card_id"], match["metadata"])
```

Omit `include_metadata=True` for the smallest recognition-only download.

</details>

<details>
<summary><strong>JavaScript example</strong></summary>

```javascript
import {
  BrowserCatalogV2,
} from "https://hanclinto.github.io/CollectorVision/lib/collectorvision-catalog-v2.mjs";

const catalog = await BrowserCatalogV2.forGame("mtg", {
  includeMetadata: true,
});

const [match] = catalog.searchRecords(queryEmbedding, 1);
console.log(match.card_id, match.metadata);
```

`queryEmbedding` is the normalized `Float32Array` produced by the Milo browser
model.

</details>

## Catalog v1 or v2?

| Catalog v1 | Catalog v2 |
| --- | --- |
| One convenient NPZ file | Compact FP16 recognition plus compressed JSONL |
| Card IDs with minimal built-in data | Recognition-ready finishes plus optional names, sets, colors, and other metadata |
| Updates replace the whole file | Updates normally download only changed rows |
| Great for custom-built catalogs | Great for hosted, bandwidth-conscious applications |

Catalog v1 remains supported and is often the simplest choice when building and
distributing your own catalog. Catalog v2 is designed for applications that
want smaller downloads, richer optional data, browser-friendly files, and
incremental updates.

## How incremental updates work

Your first installation downloads a complete catalog. Later updates normally
download only cards that changed, with automatic full refreshes when needed.
Each game and optional metadata package updates independently.

[Learn how catalog updates work](docs/catalog-updates.md).

Contributor setup and architecture are documented in
[DEVELOPMENT.md](DEVELOPMENT.md).
