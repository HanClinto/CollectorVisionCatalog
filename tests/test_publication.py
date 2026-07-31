from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest
from conftest import TrackingEmbedder, TrackingImageLoader, build_test_catalog, make_row

from collectorvision_catalog import (
    CatalogVersionManifest,
    ValidationError,
    plan_catalog_version,
    publish_catalog_version,
)


def _build(
    workspace: Path,
    version: int,
    previous=None,
):
    build_dir = workspace / f"builder-{version}"
    rows = [
        make_row(
            "alpha",
            "memory://alpha",
            f"fp-{version}",
            metadata={"name": "Alpha", "version": version},
        )
    ]
    build = build_test_catalog(
        rows,
        embedder=TrackingEmbedder(),
        image_loader=TrackingImageLoader({"memory://alpha": (255, version, 0)}),
        output_dir=build_dir,
        catalog_key="milo1/scryfall/mtg",
        version=str(version),
        embedding_model="milo1",
        previous_build=previous,
    )
    return build, build_dir


def test_initial_version_publishes_only_readable_base_paths(workspace: Path) -> None:
    build, build_dir = _build(workspace, 0)

    manifest, path = publish_catalog_version(
        build,
        build_dir,
        workspace / "public",
        "scryfall-mtg",
        plan_catalog_version(None),
    )

    assert path == workspace / "public/scryfall-mtg/version/0"
    assert not (path / "manifest.json").exists()
    assert manifest.delta is None
    assert manifest.base is not None
    assert manifest.base.rows == 1
    assert set(manifest.base.assets) == {"records", "embeddings"}
    assert {
        asset.path for asset in manifest.base.assets.values()
    } == {
        "base/embeddings.f16.gz",
        "base/records.jsonl.gz",
    }
    for asset in manifest.base.assets.values():
        assert (path / asset.path).is_file()
    with gzip.open(path / "base/records.jsonl.gz", "rt") as stream:
        records = [json.loads(line) for line in stream]
    assert records
    assert all("metadata" in record for record in records)
    assert CatalogVersionManifest.from_dict(manifest.to_dict()) == manifest


def test_public_manifest_requires_positive_base_rows(workspace: Path) -> None:
    build, build_dir = _build(workspace, 0)
    manifest, _ = publish_catalog_version(
        build,
        build_dir,
        workspace / "public",
        "scryfall-mtg",
        plan_catalog_version(None),
    )
    payload = manifest.to_dict()
    payload["base"]["rows"] = 0

    with pytest.raises(ValidationError, match="base.rows must be a positive integer"):
        CatalogVersionManifest.from_dict(payload)


def test_public_manifest_base_rows_match_catalog_rows(workspace: Path) -> None:
    build, build_dir = _build(workspace, 0)
    manifest, _ = publish_catalog_version(
        build,
        build_dir,
        workspace / "public",
        "scryfall-mtg",
        plan_catalog_version(None),
    )
    payload = manifest.to_dict()
    payload["base"]["rows"] += 1

    with pytest.raises(ValidationError, match="base.rows must match manifest rows"):
        CatalogVersionManifest.from_dict(payload)


def test_public_manifest_rejects_removed_asset_rows(workspace: Path) -> None:
    build, build_dir = _build(workspace, 0)
    manifest, _ = publish_catalog_version(
        build,
        build_dir,
        workspace / "public",
        "scryfall-mtg",
        plan_catalog_version(None),
    )
    payload = manifest.to_dict()
    payload["base"]["assets"]["records"]["rows"] = 1

    with pytest.raises(ValidationError, match="published asset fields"):
        CatalogVersionManifest.from_dict(payload)


def test_incremental_version_publishes_only_delta(workspace: Path) -> None:
    previous, _ = _build(workspace, 0)
    build, build_dir = _build(workspace, 1, previous)

    manifest, path = publish_catalog_version(
        build,
        build_dir,
        workspace / "public",
        "scryfall-mtg",
        plan_catalog_version(0),
        previous_build=previous,
    )

    assert manifest.base is None
    assert manifest.delta is not None
    assert manifest.delta.from_version == 0
    assert manifest.delta.rows.to_dict() == {
        "added": 0,
        "updated": 1,
        "deleted": 0,
    }
    assert manifest.delta.recognition_rows == 1
    assert manifest.delta.metadata_rows == 1
    assert set(manifest.delta.assets) == {"records", "embeddings"}
    assert not (path / "base").exists()
    assert (path / "delta-from-0/embeddings.f16.gz").is_file()
    with gzip.open(path / "delta-from-0/records.jsonl.gz", "rt") as stream:
        operations = [json.loads(line) for line in stream]
    assert operations
    assert all(operation["op"] == "upsert" for operation in operations)
    assert all("state" not in operation["record"] for operation in operations)


def test_routine_and_hard_checkpoints_have_distinct_routes(workspace: Path) -> None:
    previous, _ = _build(workspace, 9)
    build, build_dir = _build(workspace, 10, previous)

    routine, _ = publish_catalog_version(
        build,
        build_dir,
        workspace / "routine",
        "scryfall-mtg",
        plan_catalog_version(9),
        previous_build=previous,
    )
    hard, _ = publish_catalog_version(
        build,
        build_dir,
        workspace / "hard",
        "scryfall-mtg",
        plan_catalog_version(9, force_full_refresh=True),
    )

    assert routine.base is not None and routine.delta is not None
    assert hard.base is not None and hard.delta is None


def test_publication_rejects_a_mismatched_plan(workspace: Path) -> None:
    build, build_dir = _build(workspace, 0)

    with pytest.raises(ValidationError, match="version does not match"):
        publish_catalog_version(
            build,
            build_dir,
            workspace / "public",
            "scryfall-mtg",
            plan_catalog_version(0),
        )


def test_delete_only_delta_does_not_require_embeddings(workspace: Path) -> None:
    previous_dir = workspace / "builder-0"
    previous = build_test_catalog(
        [
            make_row("alpha", "memory://alpha", "fp-alpha"),
            make_row("beta", "memory://beta", "fp-beta"),
        ],
        embedder=TrackingEmbedder(),
        image_loader=TrackingImageLoader(
            {"memory://alpha": (255, 0, 0), "memory://beta": (0, 255, 0)}
        ),
        output_dir=previous_dir,
        catalog_key="milo1/scryfall/mtg",
        version="0",
        embedding_model="milo1",
    )
    build_dir = workspace / "builder-1"
    build = build_test_catalog(
        [make_row("alpha", "memory://alpha", "fp-alpha")],
        embedder=TrackingEmbedder(),
        image_loader=TrackingImageLoader({"memory://alpha": (255, 0, 0)}),
        output_dir=build_dir,
        catalog_key="milo1/scryfall/mtg",
        version="1",
        embedding_model="milo1",
        previous_build=previous,
    )

    manifest, path = publish_catalog_version(
        build,
        build_dir,
        workspace / "public",
        "scryfall-mtg",
        plan_catalog_version(0),
        previous_build=previous,
    )

    assert manifest.delta is not None
    assert manifest.delta.rows.to_dict() == {
        "added": 0,
        "updated": 0,
        "deleted": 1,
    }
    assert manifest.delta.recognition_rows == 1
    assert manifest.delta.metadata_rows == 0
    assert set(manifest.delta.assets) == {"records"}
    assert CatalogVersionManifest.from_dict(manifest.to_dict()) == manifest
    with gzip.open(path / "delta-from-0/records.jsonl.gz", "rt") as stream:
        operations = [json.loads(line) for line in stream]
    assert operations == [{"op": "delete", "id": "beta"}]


def test_metadata_only_delta_omits_recognition_assets(workspace: Path) -> None:
    previous, _ = _build(workspace, 0)
    build_dir = workspace / "builder-1"
    build = build_test_catalog(
        [
            make_row(
                "alpha",
                "memory://alpha",
                "fp-0",
                name="Alpha",
                metadata={"set": "Renamed"},
            )
        ],
        embedder=TrackingEmbedder(),
        image_loader=TrackingImageLoader({"memory://alpha": (255, 0, 0)}),
        output_dir=build_dir,
        catalog_key="milo1/scryfall/mtg",
        version="1",
        embedding_model="milo1",
        previous_build=previous,
    )

    manifest, path = publish_catalog_version(
        build,
        build_dir,
        workspace / "public",
        "scryfall-mtg",
        plan_catalog_version(0),
        previous_build=previous,
    )

    assert manifest.delta is not None
    assert manifest.delta.rows.to_dict() == {
        "added": 0,
        "updated": 1,
        "deleted": 0,
    }
    assert manifest.delta.recognition_rows == 0
    assert manifest.delta.metadata_rows == 1
    assert set(manifest.delta.assets) == {"records"}
    assert CatalogVersionManifest.from_dict(manifest.to_dict()) == manifest
    with gzip.open(path / "delta-from-0/records.jsonl.gz", "rt") as stream:
        operations = [json.loads(line) for line in stream]
    assert len(operations) == 1
    assert operations[0]["op"] == "upsert"
    assert operations[0]["metadata"] == {"set": "Renamed"}
    assert "embedding_index" not in operations[0]


def test_publication_rejects_tampered_builder_asset(workspace: Path) -> None:
    build, build_dir = _build(workspace, 0)
    (build_dir / build.manifest.assets["embeddings"].filename).write_bytes(b"tampered")

    with pytest.raises(ValidationError, match="failed integrity validation"):
        publish_catalog_version(
            build,
            build_dir,
            workspace / "public",
            "scryfall-mtg",
            plan_catalog_version(None),
        )


def test_publication_does_not_overwrite_an_immutable_version(workspace: Path) -> None:
    build, build_dir = _build(workspace, 0)
    output_root = workspace / "public"
    publish_catalog_version(
        build,
        build_dir,
        output_root,
        "scryfall-mtg",
        plan_catalog_version(None),
    )

    with pytest.raises(ValidationError, match="already exists"):
        publish_catalog_version(
            build,
            build_dir,
            output_root,
            "scryfall-mtg",
            plan_catalog_version(None),
        )
