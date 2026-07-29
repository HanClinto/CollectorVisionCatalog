from __future__ import annotations

from pathlib import Path

import pytest
from conftest import TrackingEmbedder, TrackingImageLoader, build_test_catalog, make_row

from collectorvision_catalog import (
    ValidationError,
    load_catalog_version_manifest,
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

    assert path == workspace / "public/scryfall-mtg/version/0/manifest.json"
    assert path.read_text().startswith('{\n  "base":')
    assert manifest.delta is None
    assert manifest.base is not None
    assert set(manifest.base) == {"embeddings", "identifiers", "metadata"}
    assert {asset.rows for asset in manifest.base.values()} == {1}
    assert {
        asset.path for asset in manifest.base.values()
    } == {
        "base/embeddings.f16.gz",
        "base/identifiers.jsonl.gz",
        "base/metadata.jsonl.gz",
    }
    for asset in manifest.base.values():
        assert (path.parent / asset.path).is_file()
    assert load_catalog_version_manifest(path) == manifest


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
    assert manifest.delta.rows == 1
    assert manifest.delta.recognition.to_dict() == {
        "added": 0,
        "updated": 1,
        "deleted": 0,
    }
    assert manifest.delta.metadata.to_dict() == {
        "added": 0,
        "updated": 1,
        "deleted": 0,
    }
    assert set(manifest.delta.assets) == {"embeddings", "identifiers", "metadata"}
    assert {asset.rows for asset in manifest.delta.assets.values()} == {1}
    assert not (path.parent / "base").exists()
    assert (path.parent / "delta-from-0/embeddings.f16.gz").is_file()


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
    assert manifest.delta.rows == 1
    assert manifest.delta.recognition.to_dict() == {
        "added": 0,
        "updated": 0,
        "deleted": 1,
    }
    assert manifest.delta.metadata.to_dict() == {
        "added": 0,
        "updated": 0,
        "deleted": 0,
    }
    assert set(manifest.delta.assets) == {"identifiers"}
    assert manifest.delta.assets["identifiers"].rows == 1
    assert load_catalog_version_manifest(path) == manifest


def test_metadata_only_delta_omits_recognition_assets(workspace: Path) -> None:
    previous, _ = _build(workspace, 0)
    build_dir = workspace / "builder-1"
    build = build_test_catalog(
        [
            make_row(
                "alpha",
                "memory://alpha",
                "fp-0",
                metadata={"name": "Renamed"},
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
    assert manifest.delta.rows == 1
    assert manifest.delta.recognition.total == 0
    assert manifest.delta.metadata.to_dict() == {
        "added": 0,
        "updated": 1,
        "deleted": 0,
    }
    assert set(manifest.delta.assets) == {"metadata"}
    assert manifest.delta.assets["metadata"].rows == 1
    assert load_catalog_version_manifest(path) == manifest


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
