from __future__ import annotations

import importlib.util
import json
import os
import sys
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError

import numpy as np
import pytest
from PIL import Image

from collectorvision_catalog import PrimaryID, RecognitionRow, ValidationError

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
        face_index=0,
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
        face_index=1,
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


def test_scryfall_cache_resolves_sharded_face_and_revision(tmp_path: Path) -> None:
    images_root = tmp_path / "scryfall" / "images" / "png"
    (images_root / "front").mkdir(parents=True)
    (images_root / "back").mkdir()
    row = make_row("https://cards.scryfall.io/png/front/c/a/card-1.png?123")
    path = images_root / "front" / "c" / "a" / "card-1.png"
    path.parent.mkdir(parents=True)
    Image.new("RGB", (2, 2), (255, 0, 0)).save(path)
    os.utime(path, (123, 123))

    cache = updater.ScryfallImageCache(tmp_path, [row])
    assert cache.path_for_row(row) == path
    assert cache.is_current(row)
    image = cache(row.image_url)
    try:
        assert image.getpixel((0, 0)) == (255, 0, 0)
    finally:
        image.close()


def test_tcgplayer_cache_resolves_sharded_product_image(tmp_path: Path) -> None:
    images_root = tmp_path / "tcgplayer" / "images" / "product"
    path = images_root / "1" / "2" / "12345.jpg"
    path.parent.mkdir(parents=True)
    Image.new("RGB", (2, 2), (255, 0, 0)).save(path)
    row = RecognitionRow(
        key="tcgplayer:12345:face:0",
        primary_id=PrimaryID("tcgplayer", "12345"),
        secondary_ids={},
        face_index=0,
        image_url="https://tcgplayer-cdn.tcgplayer.com/product/12345_in_1000x1000.jpg",
        image_fingerprint="fingerprint",
        metadata={"name": "Card"},
    )

    cache = updater.TCGplayerImageCache(tmp_path, [row])
    assert cache.path_for_row(row) == path
    assert cache.is_cached(row)
    image = cache(row.image_url)
    try:
        assert image.getpixel((0, 0)) == (254, 0, 0)
    finally:
        image.close()

    path.write_bytes(b"")
    assert not cache.is_cached(row)


def test_tcgplayer_download_falls_back_to_second_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    requested: list[str] = []

    def fake_urlopen(request: object, timeout: int) -> BytesIO:
        url = request.full_url
        requested.append(url)
        if "_1_in_" in url:
            raise HTTPError(url, 403, "Forbidden", {}, None)
        return BytesIO(b"alternate")

    monkeypatch.setattr(updater, "urlopen", fake_urlopen)
    payload = updater._download_tcgplayer_image(
        "https://tcgplayer-cdn.tcgplayer.com/product/123_1_in_1000x1000.jpg",
        attempts=1,
    )
    assert payload == b"alternate"
    assert requested == [
        "https://tcgplayer-cdn.tcgplayer.com/product/123_1_in_1000x1000.jpg",
        "https://tcgplayer-cdn.tcgplayer.com/product/123_2_in_1000x1000.jpg",
    ]


def test_tcgplayer_download_falls_back_to_first_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    requested: list[str] = []

    def fake_urlopen(request: object, timeout: int) -> BytesIO:
        url = request.full_url
        requested.append(url)
        if "_1_in_" not in url:
            raise HTTPError(url, 403, "Forbidden", {}, None)
        return BytesIO(b"alternate")

    monkeypatch.setattr(updater, "urlopen", fake_urlopen)
    payload = updater._download_tcgplayer_image(
        "https://tcgplayer-cdn.tcgplayer.com/product/123_in_1000x1000.jpg",
        attempts=1,
    )
    assert payload == b"alternate"
    assert requested == [
        "https://tcgplayer-cdn.tcgplayer.com/product/123_in_1000x1000.jpg",
        "https://tcgplayer-cdn.tcgplayer.com/product/123_1_in_1000x1000.jpg",
    ]


def test_changed_row_budget_counts_updates_additions_and_removals() -> None:
    alpha = make_row()
    beta = replace(alpha, key="scryfall:card-2:face:0")
    previous = SimpleNamespace(rows=(alpha, beta))
    changed_alpha = replace(alpha, image_fingerprint="changed")
    added = replace(alpha, key="scryfall:card-3:face:0")
    assert updater._count_changed_image_rows([changed_alpha, added], previous) == 3


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
    quality_report = json.loads((output_dir / "quality-report.json").read_text())
    assert quality_report["catalogs"]["milo1/scryfall/mtg"]["excluded_rows"] == 0
