from __future__ import annotations

import importlib.util
import json
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import TrackingEmbedder, TrackingImageLoader, make_row
from conftest import build_test_catalog as build_catalog

from collectorvision_catalog import (
    AssetIntegrityError,
    SourceRevision,
    ValidationError,
    assemble_seed_release,
    load_catalog_index,
    manifest_filename_for_catalog,
    validate_release,
    write_catalog_index,
    write_checksums,
)

VERSION = "catalog-v2-beta.1-2026-07-24"
SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "assemble_release.py"
SPEC = importlib.util.spec_from_file_location("assemble_release", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
ASSEMBLE_RELEASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ASSEMBLE_RELEASE)


def _seed(directory: Path, catalog_key: str, *, version: str = VERSION) -> Path:
    row_key = catalog_key.rsplit("/", 1)[-1]
    build_catalog(
        [make_row(row_key, f"memory://{row_key}", f"fp-{row_key}")],
        embedder=TrackingEmbedder(),
        image_loader=TrackingImageLoader({f"memory://{row_key}": (255, 0, 0)}),
        output_dir=directory,
        catalog_key=catalog_key,
        version=version,
        embedding_model="milo1",
    )
    manifest = directory / manifest_filename_for_catalog(catalog_key)
    write_catalog_index(directory / "catalog-index-v2.json", version, {catalog_key: manifest})
    (directory / "quality-report.json").write_text(
        json.dumps({"version": version, "catalogs": {catalog_key: {"excluded_rows": 0}}}),
        encoding="utf-8",
    )
    (directory / "seed-summary.json").write_text(
        json.dumps({"version": version, "catalogs": [{"catalog_key": catalog_key}]}),
        encoding="utf-8",
    )
    return manifest


def test_assembles_seed_release_atomically_and_preserves_summaries(workspace: Path) -> None:
    first = workspace / "scryfall-seed"
    second = workspace / "tcgplayer-seed"
    _seed(first, "milo1/scryfall/mtg")
    _seed(second, "milo1/tcgplayer/pokemon")
    output = workspace / "release"

    assembled = assemble_seed_release([first, second], output, VERSION)

    assert sorted(assembled.catalogs) == [
        "milo1/scryfall/mtg",
        "milo1/tcgplayer/pokemon",
    ]
    assert validate_release(output, expected_version=VERSION) == assembled
    assert all(path.is_file() for path in output.iterdir())
    summary = json.loads((output / "seed-summary.json").read_text())
    assert [item["catalog_keys"] for item in summary["inputs"]] == [
        ["milo1/scryfall/mtg"],
        ["milo1/tcgplayer/pokemon"],
    ]
    assert summary["inputs"][0]["summary"]["version"] == VERSION
    quality = json.loads((output / "quality-report.json").read_text())
    assert sorted(quality["catalogs"]) == sorted(assembled.catalogs)


def test_assembly_rejects_existing_output_duplicate_key_and_version(workspace: Path) -> None:
    first = workspace / "first"
    second = workspace / "second"
    _seed(first, "demo/catalog")
    _seed(second, "demo/catalog")
    existing = workspace / "existing"
    existing.mkdir()

    with pytest.raises(ValidationError, match="already exists"):
        assemble_seed_release([first], existing, VERSION)
    with pytest.raises(ValidationError, match="duplicate catalog key"):
        assemble_seed_release([first, second], workspace / "duplicate", VERSION)
    with pytest.raises(ValidationError, match="does not match expected"):
        assemble_seed_release([first], workspace / "wrong-version", "other-version")
    assert not (workspace / "duplicate").exists()
    assert not (workspace / "wrong-version").exists()


def test_assembly_rejects_filename_collision_and_nested_asset(workspace: Path) -> None:
    first = workspace / "first"
    second = workspace / "second"
    _seed(first, "demo/first")
    _seed(second, "demo/second")
    first_index = load_catalog_index(first / "catalog-index-v2.json")
    second_payload = json.loads((second / "catalog-index-v2.json").read_text())
    second_entry = second_payload["catalogs"]["demo/second"]
    second_entry["manifest_filename"] = first_index.catalogs["demo/first"].manifest_filename
    (second / "catalog-index-v2.json").write_text(json.dumps(second_payload), encoding="utf-8")

    with pytest.raises(ValidationError, match="cannot read manifest|filename collision"):
        assemble_seed_release([first, second], workspace / "collision", VERSION)

    manifest = _seed(workspace / "nested", "demo/nested")
    payload = json.loads(manifest.read_text())
    payload["assets"]["recognition_rows"]["filename"] = "../rows.jsonl.gz"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    index_payload = json.loads((manifest.parent / "catalog-index-v2.json").read_text())
    entry = index_payload["catalogs"]["demo/nested"]
    entry["sha256"] = sha256(manifest.read_bytes()).hexdigest()
    (manifest.parent / "catalog-index-v2.json").write_text(
        json.dumps(index_payload), encoding="utf-8"
    )
    with pytest.raises(ValidationError, match="flat filename"):
        assemble_seed_release([manifest.parent], workspace / "path", VERSION)


def test_assembly_rejects_tampered_manifest_asset_and_non_seed(workspace: Path) -> None:
    manifest_seed = workspace / "manifest-tampered"
    manifest = _seed(manifest_seed, "demo/manifest")
    manifest.write_bytes(manifest.read_bytes() + b" ")
    with pytest.raises(ValidationError, match="manifest checksum mismatch"):
        assemble_seed_release([manifest_seed], workspace / "bad-manifest", VERSION)

    asset_seed = workspace / "asset-tampered"
    asset_manifest = _seed(asset_seed, "demo/asset")
    asset_payload = json.loads(asset_manifest.read_text())
    asset_path = asset_seed / asset_payload["assets"]["recognition_rows"]["filename"]
    asset_path.write_bytes(asset_path.read_bytes() + b"tampered")
    with pytest.raises(AssetIntegrityError, match="checksum|size"):
        assemble_seed_release([asset_seed], workspace / "bad-asset", VERSION)

    previous_dir = workspace / "previous"
    previous_manifest = _seed(previous_dir, "demo/incremental")
    from collectorvision_catalog import load_catalog_build

    previous = load_catalog_build(previous_manifest)
    incremental = workspace / "incremental"
    build_catalog(
        [make_row("incremental", "memory://incremental", "fp-incremental")],
        embedder=TrackingEmbedder(),
        image_loader=TrackingImageLoader({"memory://incremental": (255, 0, 0)}),
        output_dir=incremental,
        catalog_key="demo/incremental",
        version=VERSION,
        embedding_model="milo1",
        previous_build=previous,
    )
    incremental_manifest = incremental / manifest_filename_for_catalog("demo/incremental")
    write_catalog_index(
        incremental / "catalog-index-v2.json",
        VERSION,
        {"demo/incremental": incremental_manifest},
    )
    (incremental / "quality-report.json").write_text(
        json.dumps(
            {"version": VERSION, "catalogs": {"demo/incremental": {"excluded_rows": 0}}}
        )
    )
    (incremental / "seed-summary.json").write_text(json.dumps({"version": VERSION}))
    with pytest.raises(ValidationError, match="seed manifest"):
        assemble_seed_release([incremental], workspace / "not-seed", VERSION)


def test_checksums_are_deterministic_and_existing_release_is_validated(workspace: Path) -> None:
    seed = workspace / "seed"
    _seed(seed, "demo/catalog")
    output = workspace / "release"
    assemble_seed_release([seed], output, VERSION)
    checksums = output / "SHA256SUMS"
    assert not checksums.exists()

    write_checksums(output)
    original = checksums.read_bytes()
    write_checksums(output)
    assert checksums.read_bytes() == original
    lines = original.decode().splitlines()
    assert lines == sorted(lines, key=lambda line: line.split("  ", 1)[1])
    assert all(not line.endswith("SHA256SUMS") for line in lines)

    quality = output / "quality-report.json"
    quality_bytes = quality.read_bytes()
    quality.write_bytes(quality_bytes + b" ")
    with pytest.raises(ValidationError, match="SHA256SUMS"):
        validate_release(output)
    quality.write_bytes(quality_bytes)

    manifest = output / manifest_filename_for_catalog("demo/catalog")
    manifest.write_bytes(manifest.read_bytes() + b" ")
    with pytest.raises(ValidationError, match="checksum"):
        validate_release(output)


def test_cli_assembles_and_validates_release(workspace: Path) -> None:
    seed = workspace / "seed"
    _seed(seed, "demo/catalog")
    output = workspace / "release"
    assert (
        ASSEMBLE_RELEASE.main(
            [
                "assemble",
                "--input-dir",
                str(seed),
                "--output-dir",
                str(output),
                "--version",
                VERSION,
            ]
        )
        == 0
    )
    checksum_path = output / "SHA256SUMS"
    assert not checksum_path.exists()
    with pytest.raises(ValidationError, match="does not match expected"):
        ASSEMBLE_RELEASE.main(
            [
                "validate",
                "--release-dir",
                str(output),
                "--version",
                "wrong-version",
            ]
        )
    assert not checksum_path.exists()
    assert (
        ASSEMBLE_RELEASE.main(
            [
                "validate",
                "--release-dir",
                str(output),
                "--version",
                VERSION,
                "--write-checksums",
            ]
        )
        == 0
    )
    assert checksum_path.is_file()


def test_beta_release_date_must_match_latest_source_date(workspace: Path) -> None:
    seed = workspace / "seed"
    _seed(seed, "demo/catalog", version="catalog-v2-beta.2-2026-07-25")
    with pytest.raises(ValidationError, match="date suffix"):
        validate_release(seed)


def test_release_index_contains_only_manifest_discovery_fields(workspace: Path) -> None:
    seed = workspace / "seed"
    _seed(seed, "demo/catalog")
    index_path = seed / "catalog-index-v2.json"
    payload = json.loads(index_path.read_text())
    assert set(payload["catalogs"]["demo/catalog"]) == {
        "manifest_filename",
        "sha256",
    }


def test_release_rejects_incorrect_top_level_source_timestamp(workspace: Path) -> None:
    seed = workspace / "seed"
    _seed(seed, "demo/catalog")
    index_path = seed / "catalog-index-v2.json"
    payload = json.loads(index_path.read_text())
    payload["source_updated_at"] = "2026-07-24T23:59:59Z"
    index_path.write_text(json.dumps(payload))
    with pytest.raises(ValidationError, match="maximum manifest source timestamp"):
        validate_release(seed)


def test_source_status_cli_reports_latest_revision(monkeypatch, capsys) -> None:
    configs = [
        SimpleNamespace(
            key="scryfall",
            enabled=True,
            source={"type": "scryfall", "bulk_type": "default_cards"},
        ),
        SimpleNamespace(key="tcgcsv", enabled=True, source={"type": "tcgcsv"}),
    ]
    scryfall = SourceRevision(
        "scryfall",
        "default_cards",
        "2026-07-24T21:11:04.682Z",
        "https://data.scryfall.io/default.jsonl.gz",
        "bulk",
    )
    tcgcsv = SourceRevision(
        "tcgcsv",
        "tcgplayer",
        "2026-07-24T20:11:00Z",
        "https://tcgcsv.com/last-updated.txt",
        "2026-07-24T20:11:00Z",
    )
    monkeypatch.setattr(
        ASSEMBLE_RELEASE.runpy,
        "run_path",
        lambda path: {
            "load_config": lambda path: configs,
            "fetch_scryfall_revision": lambda source: scryfall,
            "fetch_tcgcsv_revision": lambda: tcgcsv,
        },
    )
    assert ASSEMBLE_RELEASE.main(["source-status"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["source_updated_at"] == scryfall.updated_at
    assert payload["suggested_date_suffix"] == "2026-07-24"
