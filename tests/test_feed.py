from __future__ import annotations

from pathlib import Path

import pytest
from conftest import TrackingEmbedder, TrackingImageLoader, build_test_catalog, make_row

from collectorvision_catalog import (
    CatalogFeed,
    CatalogVersionPlan,
    ValidationError,
    load_catalog_feed,
    plan_catalog_version,
    publish_catalog_version,
    update_catalog_feed,
    write_catalog_feed,
)

CATALOG_KEY = "milo1/scryfall/mtg"


def _build(workspace: Path, version: int, previous=None):
    directory = workspace / f"build-{version}"
    build = build_test_catalog(
        [
            make_row(
                "alpha",
                "memory://alpha",
                f"fp-{version}",
                metadata={"name": f"Alpha {version}"},
            )
        ],
        embedder=TrackingEmbedder(),
        image_loader=TrackingImageLoader({"memory://alpha": (255, version % 256, 0)}),
        output_dir=directory,
        catalog_key=CATALOG_KEY,
        version=str(version),
        embedding_model="milo1",
        previous_build=previous,
    )
    return build, directory


def _publish(workspace: Path, version: int, previous=None, *, hard: bool = False):
    build, directory = _build(workspace, version, previous)
    previous_version = None if previous is None else int(previous.manifest.version)
    plan = (
        plan_catalog_version(previous_version, force_full_refresh=hard)
        if previous_version is None or version == previous_version + 1
        else CatalogVersionPlan(version, previous_version, True, False)
    )
    manifest, path = publish_catalog_version(
        build,
        directory,
        workspace / "public",
        "scryfall-mtg",
        plan,
        previous_build=previous,
    )
    return build, (path, manifest)


def _feed(records, checked_at: str = "2026-07-29T20:00:00Z"):
    return update_catalog_feed({CATALOG_KEY: records}, checked_at=checked_at)


def test_initial_base_only_feed(workspace: Path) -> None:
    _, record = _publish(workspace, 0)

    feed = _feed([record])
    entry = feed.catalogs[CATALOG_KEY]

    assert feed.to_dict().keys() == {"checked_at", "catalogs"}
    assert entry.public_name == "scryfall-mtg"
    assert entry.current_version == 0
    assert entry.base.version == 0
    assert set(entry.base.assets) == {"embeddings", "identifiers", "metadata"}
    assert entry.deltas == ()
    assert "/scryfall-mtg/version/0/manifest.json" in entry.base.manifest.url
    assert set(entry.base.manifest.to_dict()) == {"url", "sha256", "size"}


def test_version_zero_delta_chain(workspace: Path) -> None:
    build0, record0 = _publish(workspace, 0)
    build1, record1 = _publish(workspace, 1, build0)
    _, record2 = _publish(workspace, 2, build1)

    entry = _feed([record0, record1, record2]).catalogs[CATALOG_KEY]

    assert entry.base.version == 0
    assert [(delta.from_version, delta.to_version) for delta in entry.deltas] == [
        (0, 1),
        (1, 2),
    ]


def test_routine_checkpoint_keeps_bridge_delta(workspace: Path) -> None:
    build9, _ = _build(workspace, 9)
    build10, record10 = _publish(workspace, 10, build9)
    _, record11 = _publish(workspace, 11, build10)

    entry = _feed([record10, record11]).catalogs[CATALOG_KEY]

    assert entry.base.version == 10
    assert [(delta.from_version, delta.to_version) for delta in entry.deltas] == [
        (9, 10),
        (10, 11),
    ]


def test_hard_checkpoint_drops_earlier_deltas(workspace: Path) -> None:
    build9, _ = _build(workspace, 9)
    build10, record10 = _publish(workspace, 10, build9, hard=True)
    _, record11 = _publish(workspace, 11, build10)

    entry = _feed([record10, record11]).catalogs[CATALOG_KEY]

    assert entry.base.version == 10
    assert [(delta.from_version, delta.to_version) for delta in entry.deltas] == [(10, 11)]


def test_invalid_history_is_rejected(workspace: Path) -> None:
    build0, record0 = _publish(workspace, 0)
    build1, _ = _publish(workspace, 1, build0)
    _, record2 = _publish(workspace, 2, build1)

    with pytest.raises(ValidationError, match="contiguous"):
        _feed([record0, record2])


def test_invalid_urls_and_checksums_are_rejected(workspace: Path) -> None:
    _, record = _publish(workspace, 0)
    payload = _feed([record]).to_dict()
    base = payload["catalogs"][CATALOG_KEY]["base"]
    base["manifest"]["url"] = "http://example.com/manifest.json"
    with pytest.raises(ValidationError, match="must be under"):
        CatalogFeed.from_dict(payload)

    payload = _feed([record]).to_dict()
    payload["catalogs"][CATALOG_KEY]["base"]["manifest"]["sha256"] = "not-a-checksum"
    with pytest.raises(ValidationError, match="lowercase hexadecimal"):
        CatalogFeed.from_dict(payload)


def test_asset_integrity_is_checked(workspace: Path) -> None:
    _, record = _publish(workspace, 0)
    asset = next(iter(record[1].base.values()))
    (record[0].parent / asset.path).write_bytes(b"corrupt")

    with pytest.raises(ValidationError, match="integrity"):
        _feed([record])


def test_deterministic_serialization_round_trip(workspace: Path) -> None:
    _, record = _publish(workspace, 0)
    feed = _feed([record])
    first = workspace / "first.json"
    second = workspace / "second.json"

    write_catalog_feed(first, feed)
    write_catalog_feed(second, load_catalog_feed(first))

    assert first.read_bytes() == second.read_bytes()
    assert load_catalog_feed(first) == feed
