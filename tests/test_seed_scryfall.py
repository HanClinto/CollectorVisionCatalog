from __future__ import annotations

import importlib.util
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from uuid import UUID

import numpy as np
import pytest
from PIL import Image

from collectorvision_catalog import RecognitionRow, ValidationError

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "seed_scryfall.py"
SPEC = importlib.util.spec_from_file_location("seed_scryfall", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
seed = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = seed
SPEC.loader.exec_module(seed)

CURRENT_ID = "aa000000-0000-0000-0000-000000000001"
STALE_ID = "aa000000-0000-0000-0000-000000000002"
MISSING_ID = "aa000000-0000-0000-0000-000000000003"


def make_row(card_id: str, revision: int) -> RecognitionRow:
    return RecognitionRow(
        key=f"scryfall:{card_id}:face:0",
        identifiers={"scryfall_card": card_id},
        face_index=0,
        image_url=f"https://cards.scryfall.io/png/front/a/b/{card_id}.png?{revision}",
        image_fingerprint=f"fp-{revision}",
        metadata={"name": card_id},
    )


def test_seed_plan_reports_current_stale_and_missing_cache_rows(tmp_path: Path) -> None:
    rows = [make_row(CURRENT_ID, 100), make_row(STALE_ID, 200), make_row(MISSING_ID, 300)]
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
        "legacy_embeddings_reused": 0,
        "embeddings_to_compute": 3,
        "cache_current": 1,
        "cache_stale": 1,
        "cache_missing": 1,
        "downloads_required": 2,
    }


def test_refresh_seed_cache_only_loads_pending_rows() -> None:
    rows = [make_row(CURRENT_ID, 100), make_row(STALE_ID, 200), make_row(MISSING_ID, 300)]
    loaded: set[str] = set()

    class Cache:
        def is_current(self, row: RecognitionRow) -> bool:
            return row.identifiers["scryfall_card"] == CURRENT_ID

        def __call__(self, image_url: str) -> Image.Image:
            loaded.add(image_url)
            return Image.new("RGB", (2, 2))

    plan = seed.SeedPlan(
        rows=tuple(rows),
        seed_embeddings={},
        inference_rows=tuple(rows),
        download_rows=tuple(rows),
        cache_current=1,
        cache_stale=1,
        cache_missing=1,
    )
    seed.refresh_seed_cache(plan, Cache(), workers=2)
    assert loaded == {rows[1].image_url, rows[2].image_url}


def test_seed_plan_deduplicates_shared_image_downloads() -> None:
    first = make_row(CURRENT_ID, 100)
    second = replace(
        make_row(STALE_ID, 100),
        image_url=first.image_url,
        image_fingerprint=first.image_fingerprint,
    )

    class EmptyCache:
        def path_for_row(self, row: RecognitionRow) -> Path:
            return Path(f"/missing/{row.key}")

        def is_current(self, row: RecognitionRow) -> bool:
            return False

    plan = seed.create_seed_plan([first, second], EmptyCache())

    assert len(plan.inference_rows) == 2
    assert len(plan.download_rows) == 1
    assert plan.downloads_required == 1


def test_legacy_scryfall_embeddings_map_front_and_back_rows(tmp_path: Path) -> None:
    front_id = "ab000000-0000-0000-0000-000000000001"
    back_id = "ab000000-0000-0000-0000-000000000002"
    catalog_path = tmp_path / "legacy.npz"
    embeddings = np.vstack(
        [
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
        ]
    )
    np.savez(
        catalog_path,
        embeddings=embeddings,
        card_ids=np.array([front_id, f"{back_id}_back"]),
        source=np.array("scryfall"),
        embedder_spec=np.array(json.dumps({"algo_key": "milo1"})),
    )

    loaded = seed.load_legacy_embeddings(catalog_path)

    assert sorted(loaded) == [
        f"scryfall:{front_id}:face:0",
        f"scryfall:{back_id}:face:1",
    ]
    assert np.array_equal(loaded[f"scryfall:{front_id}:face:0"], embeddings[0])


def test_legacy_scryfall_embeddings_reject_unsafe_ids(tmp_path: Path) -> None:
    catalog_path = tmp_path / "unsafe.npz"
    np.savez(
        catalog_path,
        embeddings=np.array([[1.0, 0.0]], dtype=np.float32),
        card_ids=np.array(["../../escape"]),
        source=np.array("scryfall"),
        embedder_spec=np.array(json.dumps({"algo_key": "milo1"})),
    )

    with pytest.raises(ValidationError, match="not a UUID"):
        seed.load_legacy_embeddings(catalog_path)


def test_legacy_scryfall_embeddings_decode_packed_uuid_rows(tmp_path: Path) -> None:
    card_id = "ac000000-0000-0000-0000-000000000001"
    catalog_path = tmp_path / "packed.npz"
    np.savez(
        catalog_path,
        embeddings=np.array([[1.0, 0.0]], dtype=np.float32),
        card_ids=np.frombuffer(UUID(card_id).bytes, dtype=np.uint8).reshape(1, 16),
        source=np.array("scryfall"),
        embedder_spec=np.array(json.dumps({"algo_key": "milo1"})),
    )

    loaded = seed.load_legacy_embeddings(catalog_path)

    assert list(loaded) == [f"scryfall:{card_id}:face:0"]
