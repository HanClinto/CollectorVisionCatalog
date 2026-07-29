from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import (
    TrackingEmbedder,
    TrackingImageLoader,
    make_row,
)
from conftest import (
    build_test_catalog as build_catalog,
)

from collectorvision_catalog import (
    SourceRevision,
    ValidationError,
    load_catalog_index,
    manifest_filename_for_catalog,
    write_catalog_index,
)


def test_catalog_index_roundtrip_is_deterministic(workspace: Path) -> None:
    first_dir = workspace / "first"
    second_dir = workspace / "second"
    build_catalog(
        [make_row("alpha", "memory://alpha", "fp-alpha", metadata={"name": "Alpha"})],
        embedder=TrackingEmbedder(),
        image_loader=TrackingImageLoader({"memory://alpha": (255, 0, 0)}),
        output_dir=first_dir,
        catalog_key="demo/catalog",
        version="v1",
        embedding_model="milo1",
        source_revision=SourceRevision(
            source_type="test",
            source_name="newer",
            updated_at="2026-07-25T00:00:00Z",
            uri="https://example.test/newer",
            identity="newer",
        ),
    )
    build_catalog(
        [make_row("beta", "memory://beta", "fp-beta", metadata={"name": "Beta"})],
        embedder=TrackingEmbedder(),
        image_loader=TrackingImageLoader({"memory://beta": (0, 255, 0)}),
        output_dir=second_dir,
        catalog_key="demo/other",
        version="v1",
        embedding_model="milo1",
    )

    index_path_a = workspace / "catalog-index-a.json"
    index_path_b = workspace / "catalog-index-b.json"
    manifests = {
        "demo/catalog": first_dir / manifest_filename_for_catalog("demo/catalog"),
        "demo/other": second_dir / manifest_filename_for_catalog("demo/other"),
    }
    index_a = write_catalog_index(index_path_a, "release-2026-07-17", manifests)
    index_b = write_catalog_index(index_path_b, "release-2026-07-17", manifests)
    loaded = load_catalog_index(index_path_a)

    assert index_path_a.read_bytes() == index_path_b.read_bytes()
    assert loaded.to_dict() == index_a.to_dict() == index_b.to_dict()
    assert loaded.catalogs["demo/catalog"].manifest_filename == "demo--catalog.manifest.json"
    assert set(loaded.catalogs["demo/catalog"].to_dict()) == {
        "manifest_filename",
        "sha256",
    }
    assert loaded.source_updated_at == "2026-07-25T00:00:00Z"
    assert all("/" not in entry.manifest_filename for entry in loaded.catalogs.values())


def test_catalog_index_loader_rejects_nested_manifest_filename(workspace: Path) -> None:
    index_path = workspace / "catalog-index.json"
    index_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "release_version": "release-2026-07-17",
                "catalogs": {
                    "demo/catalog": {
                        "manifest_filename": "nested/demo.manifest.json",
                        "sha256": "abc",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="flat filename"):
        load_catalog_index(index_path)
