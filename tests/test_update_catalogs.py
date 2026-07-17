from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from collectorvision_catalog import Face, PrimaryID, RecognitionRow, ValidationError

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "update_catalogs.py"
SPEC = importlib.util.spec_from_file_location("update_catalogs", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
updater = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = updater
SPEC.loader.exec_module(updater)


def make_config(path: Path, *, duplicate: bool = False) -> None:
    catalog = {
        "key": "milo1/scryfall/mtg",
        "source": {"type": "scryfall", "bulk_type": "default_cards", "languages": ["en"]},
        "embedding_model": updater.MILO1_MODEL_ID,
        "enabled": True,
        "seed_required": True,
    }
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "catalogs": [catalog, dict(catalog)] if duplicate else [catalog],
            }
        ),
        encoding="utf-8",
    )


def make_row(image_url: str = "memory://front") -> RecognitionRow:
    return RecognitionRow(
        key="scryfall:card-1:face:0",
        primary_id=PrimaryID("scryfall", "card-1"),
        secondary_ids={"scryfall_oracle": "oracle-1"},
        face=Face(index=0, name="Card", is_back=False),
        image_url=image_url,
        image_fingerprint="fingerprint-1",
        metadata={"name": "Card"},
    )


def test_load_config_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "catalogs.json"
    make_config(path, duplicate=True)
    with pytest.raises(ValidationError, match="duplicate catalog key"):
        updater.load_config(path)


def test_local_first_image_loader_uses_front_and_back_cache_names(tmp_path: Path) -> None:
    Image.new("RGB", (2, 2), (255, 0, 0)).save(tmp_path / "card-1.png")
    Image.new("RGB", (2, 2), (0, 0, 255)).save(tmp_path / "card-1_back.jpg")
    front = make_row("https://example.test/front")
    back = RecognitionRow(
        key="scryfall:card-1:face:1",
        primary_id=front.primary_id,
        secondary_ids=front.secondary_ids,
        face=Face(index=1, name="Back", is_back=True),
        image_url="https://example.test/back",
        image_fingerprint="fingerprint-2",
        metadata={"name": "Card"},
    )
    loader = updater.local_first_image_loader([front, back], [tmp_path])
    front_image = loader(front.image_url)
    back_image = loader(back.image_url)
    try:
        assert front_image.getpixel((0, 0)) == (255, 0, 0)
        assert back_image.getpixel((0, 0)) == (0, 0, 254)
    finally:
        front_image.close()
        back_image.close()


def test_build_requires_seed_unless_full_rebuild_is_explicit(tmp_path: Path) -> None:
    config_path = tmp_path / "catalogs.json"
    make_config(config_path)
    with pytest.raises(ValidationError, match="requires a seed release"):
        updater.build_enabled_catalogs(
            config_path=config_path,
            previous_dir=tmp_path / "previous",
            output_dir=tmp_path / "release",
            version="catalog-v2-2026-07-17",
            source_rows_factory=lambda source: [make_row()],
            embedder_factory=lambda model, batch: lambda images: np.array([[1.0, 0.0]]),
            image_loader=lambda url: Image.new("RGB", (2, 2)),
        )


def test_full_build_writes_release_index_and_summary(tmp_path: Path) -> None:
    config_path = tmp_path / "catalogs.json"
    make_config(config_path)
    output_dir = tmp_path / "release"
    summary = updater.build_enabled_catalogs(
        config_path=config_path,
        previous_dir=tmp_path / "previous",
        output_dir=output_dir,
        version="catalog-v2-2026-07-17",
        allow_full_rebuild=True,
        source_rows_factory=lambda source: [make_row()],
        embedder_factory=lambda model, batch: lambda images: np.array([[1.0, 0.0]]),
        image_loader=lambda url: Image.new("RGB", (2, 2)),
    )
    index = json.loads((output_dir / "catalog-index-v2.json").read_text(encoding="utf-8"))
    assert summary["changed"] is True
    assert index["catalogs"]["milo1/scryfall/mtg"]["manifest_filename"] == (
        "milo1--scryfall--mtg.manifest.json"
    )
    assert json.loads((output_dir / "update-summary.json").read_text()) == summary
