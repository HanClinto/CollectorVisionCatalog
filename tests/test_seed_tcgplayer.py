from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

from collectorvision_catalog import PrimaryID, RecognitionRow

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "seed_tcgplayer.py"
SPEC = importlib.util.spec_from_file_location("seed_tcgplayer", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
seed = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = seed
SPEC.loader.exec_module(seed)


def make_row(product_id: str, modified_on: str, face_index: int = 0) -> RecognitionRow:
    return RecognitionRow(
        key=f"tcgplayer:{product_id}:face:{face_index}",
        primary_id=PrimaryID("tcgplayer", product_id),
        secondary_ids={},
        face_index=face_index,
        image_url=(
            f"https://tcgplayer-cdn.tcgplayer.com/product/{product_id}"
            f"{'' if face_index == 0 else f'_{face_index}'}_in_1000x1000.jpg"
        ),
        image_fingerprint=f"fp-{modified_on}",
        metadata={"name": product_id, "modified_on": modified_on},
    )


def test_seed_plan_reuses_all_legacy_fronts() -> None:
    rows = [
        make_row("old", "2026-05-01T00:00:00"),
        make_row("changed", "2026-05-08T00:00:00"),
        make_row("old", "2026-05-01T00:00:00", face_index=1),
        make_row("new", "2026-05-01T00:00:00"),
    ]
    embeddings = {
        "tcgplayer:old:face:0": np.ones(128, dtype=np.float32),
        "tcgplayer:changed:face:0": np.ones(128, dtype=np.float32),
    }

    class Cache:
        @staticmethod
        def is_cached(row: RecognitionRow) -> bool:
            return row.primary_id.value != "new"

    plan = seed.create_seed_plan(
        "milo1/tcgplayer/test",
        rows,
        embeddings,
        Cache(),
    )

    assert set(plan.seed_embeddings) == {
        "tcgplayer:old:face:0",
        "tcgplayer:changed:face:0",
    }
    assert [row.key for row in plan.inference_rows] == [
        "tcgplayer:old:face:1",
        "tcgplayer:new:face:0",
    ]
    assert plan.summary()["downloads_required"] == 1
    assert plan.summary()["embeddings_to_compute"] == 2
