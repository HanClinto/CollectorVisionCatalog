from __future__ import annotations

from pathlib import Path

from conftest import TrackingEmbedder, TrackingImageLoader, make_row
from conftest import build_test_catalog as build_catalog

from collectorvision_catalog import (
    CatalogBuild,
    CatalogFeed,
    load_catalog_feed,
    manifest_filename_for_catalog,
    update_catalog_feed,
    write_catalog_feed,
    write_catalog_index,
)

CATALOG_KEY = "demo/catalog"


def _release(
    workspace: Path,
    version: str,
    name: str,
    previous: CatalogBuild | None,
) -> tuple[CatalogBuild, object]:
    directory = workspace / version
    build = build_catalog(
        [
            make_row(
                "alpha",
                "memory://alpha",
                "fp-alpha",
                metadata={"name": name},
            )
        ],
        embedder=TrackingEmbedder(),
        image_loader=TrackingImageLoader({"memory://alpha": (255, 0, 0)}),
        output_dir=directory,
        catalog_key=CATALOG_KEY,
        version=version,
        embedding_model="milo1",
        previous_build=previous,
    )
    index = write_catalog_index(
        directory / "catalog-index-v2.json",
        version,
        {CATALOG_KEY: directory / manifest_filename_for_catalog(CATALOG_KEY)},
    )
    return build, index


def test_feed_tracks_ordered_deltas_and_rolls_checkpoint(workspace: Path) -> None:
    previous_build = None
    previous_index = None
    feed: CatalogFeed | None = None
    for number in range(1, 7):
        version = f"v{number}"
        build, index = _release(
            workspace,
            version,
            f"Alpha {number}",
            previous_build,
        )
        feed = update_catalog_feed(
            current_index=index,
            current_manifests={CATALOG_KEY: build.manifest},
            previous_index=previous_index,
            previous_feed=feed,
        )
        entry = feed.catalogs[CATALOG_KEY]
        if number == 5:
            assert entry.base.version == "v1"
            assert [(delta.from_version, delta.to_version) for delta in entry.deltas] == [
                ("v1", "v2"),
                ("v2", "v3"),
                ("v3", "v4"),
                ("v4", "v5"),
            ]
        previous_build = build
        previous_index = index

    assert feed is not None
    entry = feed.catalogs[CATALOG_KEY]
    assert entry.base.version == "v6"
    assert entry.deltas == ()

    path = workspace / "catalog-feed-v2.json"
    write_catalog_feed(path, feed)
    assert load_catalog_feed(path) == feed


def test_unchanged_catalog_keeps_prior_feed_position(workspace: Path) -> None:
    first, first_index = _release(workspace, "v1", "Alpha", None)
    first_feed = update_catalog_feed(
        current_index=first_index,
        current_manifests={CATALOG_KEY: first.manifest},
    )
    second, second_index = _release(workspace, "v2", "Alpha", first)

    current = update_catalog_feed(
        current_index=second_index,
        current_manifests={CATALOG_KEY: second.manifest},
        previous_index=first_index,
        previous_feed=first_feed,
    )

    assert current.release_version == "v2"
    assert current.catalogs[CATALOG_KEY] == first_feed.catalogs[CATALOG_KEY]
