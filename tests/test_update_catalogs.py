from __future__ import annotations

import gzip
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

from collectorvision_catalog import RecognitionRow, ValidationError

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "update_catalogs.py"
SPEC = importlib.util.spec_from_file_location("update_catalogs", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
updater = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = updater
SPEC.loader.exec_module(updater)

CARD_ID = "ca000000-0000-0000-0000-000000000001"
ORACLE_ID = "0a000000-0000-0000-0000-000000000001"


def make_config(path: Path, *, duplicate: bool = False) -> None:
    catalog = {
        "key": "milo1/scryfall/mtg",
        "descriptor": {
            "game": "magic-the-gathering",
            "source": "scryfall",
            "profile": "printings",
            "description": "Test printings.",
            "result_identifier": "scryfall_card",
            "recommended": True,
        },
        "source": {"type": "scryfall", "bulk_type": "default_cards", "languages": ["en"]},
        "embedding_model": updater.MILO1_MODEL_ID,
        "enabled": True,
        "seed_required": True,
    }
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "catalogs": [catalog, dict(catalog)] if duplicate else [catalog],
            }
        ),
        encoding="utf-8",
    )


def make_row(image_url: str = "memory://front") -> RecognitionRow:
    return RecognitionRow(
        provider="scryfall",
        id=CARD_ID,
        identifiers={"scryfall_oracle": ORACLE_ID},
        image_url=image_url,
        image_fingerprint="fingerprint-1",
        metadata={"name": "Card"},
    )


def test_load_config_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "catalogs.json"
    make_config(path, duplicate=True)
    with pytest.raises(ValidationError, match="duplicate catalog key"):
        updater.load_config(path)


def test_repository_config_exposes_one_default_scryfall_catalog() -> None:
    configs = updater.load_config(Path("config/catalogs.json"))
    scryfall = [item for item in configs if item.descriptor.source == "scryfall"]
    assert len(scryfall) == 1
    assert scryfall[0].key == "milo1/scryfall/mtg"
    assert scryfall[0].descriptor.profile == "default"
    assert scryfall[0].enabled
    assert scryfall[0].seed_required


def test_scryfall_revision_is_extracted_from_selected_bulk_entry(monkeypatch) -> None:
    monkeypatch.setattr(
        updater,
        "_read_json_url",
        lambda url: {
            "data": [
                {
                    "id": "bulk-id",
                    "type": "default_cards",
                    "updated_at": "2026-07-24T21:11:04.682+00:00",
                    "jsonl_download_uri": "https://data.scryfall.io/default.jsonl.gz",
                }
            ]
        },
    )
    revision = updater.fetch_scryfall_revision({"bulk_type": "default_cards"})
    assert revision.to_dict() == {
        "type": "scryfall",
        "name": "default_cards",
        "updated_at": "2026-07-24T21:11:04.682Z",
        "uri": "https://data.scryfall.io/default.jsonl.gz",
        "identity": "bulk-id",
    }


@pytest.mark.parametrize(
    ("filename", "payload", "bulk_format"),
    [
        ("archived.json", b'[{"id":"old-card"}]', None),
        ("archived.jsonl.gz", gzip.compress(b'{"id":"old-card"}\n', mtime=0), None),
        ("archived.data", gzip.compress(b'{"id":"old-card"}\n', mtime=0), "jsonl"),
    ],
)
def test_scryfall_snapshot_accepts_archived_bulk_files(
    tmp_path: Path,
    monkeypatch,
    filename: str,
    payload: bytes,
    bulk_format: str | None,
) -> None:
    path = tmp_path / filename
    path.write_bytes(payload)
    monkeypatch.setattr(updater, "normalize_scryfall_card", lambda card: [make_row()])

    source = {
        "type": "scryfall",
        "bulk_type": "default_cards",
        "bulk_uri": str(path),
        "bulk_updated_at": "2026-07-20T12:34:56Z",
        "bulk_identity": "archived-test",
    }
    if bulk_format is not None:
        source["bulk_format"] = bulk_format
    snapshot = updater.fetch_scryfall_snapshot(source)

    assert snapshot.revision.uri == path.as_uri()
    assert snapshot.revision.updated_at == "2026-07-20T12:34:56Z"
    assert snapshot.revision.identity == "archived-test"
    assert snapshot.rows == (make_row(),)


def test_scryfall_bulk_cli_requires_timestamp() -> None:
    args = SimpleNamespace(
        scryfall_bulk_uri="archive.json",
        scryfall_bulk_updated_at=None,
        scryfall_bulk_identity=None,
        scryfall_bulk_format=None,
    )

    with pytest.raises(ValidationError, match="updated-at"):
        updater._scryfall_source_override(args)


def test_scryfall_jsonl_rejects_oversized_values(monkeypatch) -> None:
    monkeypatch.setattr(updater, "_MAX_JSON_VALUE_CHARS", 16)

    with pytest.raises(ValidationError, match="streaming size limit"):
        list(updater._iter_jsonl_objects(BytesIO(b'{"value":"too long"}\n'), "test JSONL"))


def test_archived_scryfall_files_build_an_exact_delta(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "catalogs.json"
    make_config(config_path)
    old_file = tmp_path / "old.jsonl"
    new_file = tmp_path / "new.jsonl"
    old_file.write_text('{"id":"old","lang":"en"}\n')
    new_file.write_text('{"id":"old","lang":"en"}\n{"id":"new","lang":"en"}\n')

    def normalize(card: dict) -> list[RecognitionRow]:
        card_id = CARD_ID if card["id"] == "old" else "ca000000-0000-0000-0000-000000000002"
        return [
            replace(
                make_row(),
                id=card_id,
                identifiers={"scryfall_oracle": ORACLE_ID},
                image_url=f"memory://{card['id']}",
                image_fingerprint=f"fingerprint-{card['id']}",
            )
        ]

    monkeypatch.setattr(updater, "normalize_scryfall_card", normalize)

    def embedder_factory(model, batch):
        return lambda images: np.tile(
            np.array([[1.0, 0.0]], dtype=np.float32),
            (len(images), 1),
        )

    def image_loader(url):
        return Image.new("RGB", (2, 2))

    base_dir = tmp_path / "base"
    updater.build_enabled_catalogs(
        config_path=config_path,
        previous_dir=tmp_path / "empty",
        output_dir=base_dir,
        version="catalog-v2-beta.100-2026-07-20",
        allow_full_rebuild=True,
        scryfall_source_override={
            "bulk_uri": str(old_file),
            "bulk_updated_at": "2026-07-20T00:00:00Z",
        },
        embedder_factory=embedder_factory,
        image_loader=image_loader,
    )

    update_dir = tmp_path / "update"
    updater.build_enabled_catalogs(
        config_path=config_path,
        previous_dir=base_dir,
        output_dir=update_dir,
        version="catalog-v2-beta.101-2026-07-21",
        scryfall_source_override={
            "bulk_uri": str(new_file),
            "bulk_updated_at": "2026-07-21T00:00:00Z",
        },
        embedder_factory=embedder_factory,
        image_loader=image_loader,
    )

    build = updater.load_catalog_build(
        update_dir / "milo1--scryfall--mtg.manifest.json",
        asset_dir=update_dir,
    )
    assert build.manifest.previous_version == "catalog-v2-beta.100-2026-07-20"
    assert build.manifest.delta.operations == 1
    assert "identifiers_delta" in build.manifest.assets
    assert "embeddings_delta" in build.manifest.assets


def test_tcgcsv_fetch_rejects_revision_change_mid_read(monkeypatch) -> None:
    timestamps = iter(
        ["2026-07-24T20:11:00+0000", "2026-07-24T20:12:00+0000"]
    )
    monkeypatch.setattr(updater, "_read_text_url", lambda url: next(timestamps))
    monkeypatch.setattr(
        updater,
        "_read_json_url",
        lambda url: {"success": True, "results": []},
    )
    monkeypatch.setattr(
        updater,
        "_fetch_tcgcsv_category_rows",
        lambda source, categories: [make_row()],
    )
    with pytest.raises(ValidationError, match="changed while"):
        updater.fetch_tcgcsv_snapshots(
            {"catalog": {"category_id": 1, "fetch_workers": 1}}
        )


def test_local_first_image_loader_uses_front_and_back_cache_names(tmp_path: Path) -> None:
    Image.new("RGB", (2, 2), (255, 0, 0)).save(tmp_path / f"{CARD_ID}.png")
    Image.new("RGB", (2, 2), (0, 0, 255)).save(tmp_path / f"{CARD_ID}_back.jpg")
    front = make_row("https://example.test/front")
    back = RecognitionRow(
        provider="scryfall",
        id=CARD_ID,
        identifiers=front.identifiers,
        image_url="https://example.test/back",
        image_fingerprint="fingerprint-2",
        face_index=1,
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
    row = make_row(f"https://cards.scryfall.io/png/front/c/a/{CARD_ID}.png?123")
    path = images_root / "front" / "c" / "a" / f"{CARD_ID}.png"
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
        provider="tcgplayer",
        id="12345",
        identifiers={},
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


def test_image_caches_create_canonical_roots(tmp_path: Path) -> None:
    scryfall_root = tmp_path / "new-scryfall-cache"
    assert updater._resolve_scryfall_images_root(scryfall_root) == (
        scryfall_root / "scryfall" / "images" / "png"
    )
    assert (scryfall_root / "scryfall" / "images" / "png" / "front").is_dir()
    assert (scryfall_root / "scryfall" / "images" / "png" / "back").is_dir()

    tcgplayer_root = tmp_path / "new-tcgplayer-cache"
    assert updater._resolve_tcgplayer_images_root(tcgplayer_root) == (
        tcgplayer_root / "tcgplayer" / "images" / "product"
    )
    assert (tcgplayer_root / "tcgplayer" / "images" / "product").is_dir()


def test_image_cache_rejects_unsafe_source_identifiers(tmp_path: Path) -> None:
    images_root = tmp_path / "scryfall" / "images" / "png"
    (images_root / "front").mkdir(parents=True)
    (images_root / "back").mkdir()
    unsafe_scryfall = replace(
        make_row(),
        id="../../escape",
    )
    with pytest.raises(ValidationError, match="invalid Scryfall card ID"):
        updater.ScryfallImageCache(tmp_path, [unsafe_scryfall]).path_for_row(
            unsafe_scryfall
        )

    unsafe_tcgplayer = RecognitionRow(
        provider="tcgplayer",
        id="../../escape",
        identifiers={},
        image_url="https://example.test/card.jpg",
        image_fingerprint="fingerprint",
    )
    with pytest.raises(ValidationError, match="invalid TCGplayer product ID"):
        updater.TCGplayerImageCache(tmp_path, [unsafe_tcgplayer]).path_for_row(
            unsafe_tcgplayer
        )

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


def test_refresh_tcgplayer_images_excludes_only_unavailable_rows() -> None:
    available = RecognitionRow(
        provider="tcgplayer",
        id="1",
        identifiers={},
        image_url="memory://available",
        image_fingerprint="available",
    )
    unavailable = replace(
        available,
        id="2",
        image_url="memory://unavailable",
    )

    class Cache:
        def __call__(self, image_url: str) -> Image.Image:
            if image_url == unavailable.image_url:
                raise updater.TCGplayerImageUnavailable("missing")
            return Image.new("RGB", (2, 2))

    assert updater.refresh_tcgplayer_images(
        [available, unavailable],
        Cache(),
        workers=2,
        catalog_key="demo/tcgplayer",
    ) == {unavailable.key}


def test_changed_row_budget_counts_updates_additions_and_removals() -> None:
    alpha = make_row()
    beta = replace(alpha, id="card-2")
    previous = SimpleNamespace(rows=(alpha, beta))
    changed_alpha = replace(alpha, image_fingerprint="changed")
    added = replace(alpha, id="card-3")
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


def test_descriptor_label_change_reuses_embeddings_and_starts_fresh_snapshot(
    tmp_path: Path,
) -> None:
    previous_dir = tmp_path / "previous"
    previous_dir.mkdir()
    row = make_row()
    extra_seed_row = replace(
        row,
        id="cc000000-0000-0000-0000-000000000001",
        image_url="memory://extra-seed",
    )
    seed_config = tmp_path / "seed-config.json"
    make_config(seed_config)
    seed_descriptor = updater.load_config(seed_config)[0].descriptor
    updater.build_catalog(
        [row, extra_seed_row],
        embedder=lambda images: np.array(
            [[0.6, 0.8], [1.0, 0.0]], dtype=np.float32
        ),
        output_dir=previous_dir,
        catalog_key="milo1/scryfall/mtg",
        version="catalog-v2-beta.1-2026-07-17",
        embedding_model=updater.MILO1_MODEL_ID,
        source_revision=updater.SourceRevision(
            source_type="scryfall",
            source_name="default_cards",
            updated_at="2000-01-01T00:00:00Z",
            uri="memory://default-cards",
            identity="default-cards",
        ),
        descriptor=seed_descriptor,
        image_loader=lambda url: Image.new("RGB", (2, 2)),
    )

    target = json.loads(seed_config.read_text(encoding="utf-8"))
    catalog = target["catalogs"][0]
    catalog["descriptor"]["profile"] = "default"
    target_config = tmp_path / "target-config.json"
    target_config.write_text(json.dumps(target), encoding="utf-8")
    changed_row = replace(
        extra_seed_row,
        image_fingerprint="fp-changed",
        image_url="memory://changed",
    )

    output_dir = tmp_path / "release"
    summary = updater.build_enabled_catalogs(
        config_path=target_config,
        previous_dir=previous_dir,
        output_dir=output_dir,
        version="catalog-v2-beta.2-2026-07-18",
        source_rows_factory=lambda source: [row, changed_row],
        embedder_factory=lambda model, batch: lambda images: np.array(
            [[0.0, 1.0]], dtype=np.float32
        ),
        image_loader=lambda url: Image.new("RGB", (2, 2)),
    )

    build = updater.load_catalog_build(
        output_dir / "milo1--scryfall--mtg.manifest.json",
        asset_dir=output_dir,
    )
    assert summary["catalogs"][0]["seed_embeddings_reused"] == 1
    assert summary["catalogs"][0]["seed_embeddings_computed"] == 1
    assert build.manifest.previous_version is None
    assert np.array_equal(
        build.embeddings,
        np.array([[0.6, 0.8], [0.0, 1.0]], dtype=np.float16),
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
    assert summary["source_updated_at"] == "2000-01-01T00:00:00Z"
    assert summary["catalogs"][0]["source_revision"]["identity"] == (
        "injected-test-source"
    )
    assert index["catalogs"]["milo1/scryfall/mtg"]["manifest_filename"] == (
        "milo1--scryfall--mtg.manifest.json"
    )
    assert json.loads((output_dir / "update-summary.json").read_text()) == summary
    quality_report = json.loads((output_dir / "quality-report.json").read_text())
    assert quality_report["catalogs"]["milo1/scryfall/mtg"]["excluded_rows"] == 0
    assert quality_report["catalogs"]["milo1/scryfall/mtg"]["source_revision"] == (
        summary["catalogs"][0]["source_revision"]
    )
