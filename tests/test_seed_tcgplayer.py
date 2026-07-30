from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

from collectorvision_catalog import RecognitionRow

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "seed_tcgplayer.py"
SPEC = importlib.util.spec_from_file_location("seed_tcgplayer", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
seed = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = seed
SPEC.loader.exec_module(seed)


def make_row(product_id: str, modified_on: str, face_index: int = 0) -> RecognitionRow:
    return RecognitionRow(
        provider="tcgplayer",
        id=product_id,
        identifiers={},
        image_url=(
            f"https://tcgplayer-cdn.tcgplayer.com/product/{product_id}"
            f"{'' if face_index == 0 else f'_{face_index}'}_in_1000x1000.jpg"
        ),
        image_fingerprint=f"fp-{modified_on}",
        face_index=face_index,
        metadata={"name": product_id},
    )


def test_seed_plan_reuses_all_legacy_fronts() -> None:
    rows = [
        make_row("old", "2026-05-01T00:00:00"),
        make_row("changed", "2026-05-08T00:00:00"),
        make_row("old", "2026-05-01T00:00:00", face_index=1),
        make_row("new", "2026-05-01T00:00:00"),
    ]
    embeddings = {
        "tcgplayer:old": np.ones(128, dtype=np.float32),
        "tcgplayer:changed": np.ones(128, dtype=np.float32),
    }

    class Cache:
        @staticmethod
        def is_cached(row: RecognitionRow) -> bool:
            return row.id != "new"

    plan = seed.create_seed_plan(
        "milo1/tcgplayer/test",
        rows,
        embeddings,
        Cache(),
    )

    assert set(plan.seed_embeddings) == {
        "tcgplayer:old",
        "tcgplayer:changed",
    }
    assert [row.key for row in plan.inference_rows] == [
        "tcgplayer:old:face:1",
        "tcgplayer:new",
    ]
    assert plan.summary()["downloads_required"] == 1
    assert plan.summary()["embeddings_to_compute"] == 2
