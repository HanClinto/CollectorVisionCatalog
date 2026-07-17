from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

from PIL import Image

from collectorvision_catalog import PrimaryID, RecognitionRow

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "seed_scryfall.py"
SPEC = importlib.util.spec_from_file_location("seed_scryfall", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
seed = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = seed
SPEC.loader.exec_module(seed)


def make_row(card_id: str, revision: int) -> RecognitionRow:
    return RecognitionRow(
        key=f"scryfall:{card_id}:face:0",
        primary_id=PrimaryID("scryfall", card_id),
        secondary_ids={},
        face_index=0,
        image_url=f"https://cards.scryfall.io/png/front/a/b/{card_id}.png?{revision}",
        image_fingerprint=f"fp-{revision}",
        metadata={"name": card_id},
    )


def test_seed_plan_reports_current_stale_and_missing_cache_rows(tmp_path: Path) -> None:
    rows = [make_row("current", 100), make_row("stale", 200), make_row("missing", 300)]
    images_root = tmp_path / "scryfall" / "images" / "png"
    (images_root / "front").mkdir(parents=True)
    (images_root / "back").mkdir()
    cache = seed.ScryfallImageCache(tmp_path, rows)

    current_path = cache.path_for_row(rows[0])
    current_path.parent.mkdir(parents=True)
    Image.new("RGB", (2, 2)).save(current_path)
    os.utime(current_path, (100, 100))
    stale_path = cache.path_for_row(rows[1])
    stale_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (2, 2)).save(stale_path)
    os.utime(stale_path, (50, 50))

    plan = seed.create_seed_plan(rows, cache)

    assert plan.summary() == {
        "current_rows": 3,
        "embeddings_to_compute": 3,
        "cache_current": 1,
        "cache_stale": 1,
        "cache_missing": 1,
        "downloads_required": 2,
    }


def test_refresh_seed_cache_only_loads_pending_rows() -> None:
    rows = [make_row("current", 100), make_row("stale", 200), make_row("missing", 300)]
    loaded: set[str] = set()

    class Cache:
        def is_current(self, row: RecognitionRow) -> bool:
            return row.primary_id.value == "current"

        def __call__(self, image_url: str) -> Image.Image:
            loaded.add(image_url)
            return Image.new("RGB", (2, 2))

    plan = seed.SeedPlan(
        rows=tuple(rows),
        cache_current=1,
        cache_stale=1,
        cache_missing=1,
    )
    seed.refresh_seed_cache(plan, Cache(), workers=2)
    assert loaded == {rows[1].image_url, rows[2].image_url}
