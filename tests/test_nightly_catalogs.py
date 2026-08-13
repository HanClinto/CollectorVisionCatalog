from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from conftest import TrackingEmbedder, TrackingImageLoader, build_test_catalog, make_row

from collectorvision_catalog import (
    load_catalog_build,
    load_catalog_feed,
    manifest_filename_for_catalog,
    plan_catalog_version,
    publish_catalog_version,
    update_catalog_feed,
)

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "nightly_catalogs.py"
SPEC = importlib.util.spec_from_file_location("nightly_catalogs", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
nightly = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = nightly
SPEC.loader.exec_module(nightly)

CATALOG_KEY = "milo1/scryfall/mtg"


def _build(directory: Path, version: int, previous=None):
    return build_test_catalog(
        [
            make_row(
                "alpha",
                "memory://alpha",
                f"fp-{version}",
                metadata={"version": version},
            )
        ],
        embedder=TrackingEmbedder(),
        image_loader=TrackingImageLoader({"memory://alpha": (version, 0, 0)}),
        output_dir=directory,
        catalog_key=CATALOG_KEY,
        version=str(version),
        embedding_model="milo1",
        previous_build=previous,
    )


def test_finalize_nightly_publishes_delta_feed_release_and_next_state(
    workspace: Path,
) -> None:
    previous_dir = workspace / "previous"
    previous = _build(previous_dir, 0)
    base_receipt, base_path = publish_catalog_version(
        previous,
        previous_dir,
        workspace / "initial-public",
        "scryfall-mtg",
        plan_catalog_version(None),
    )
    feed = update_catalog_feed(
        {CATALOG_KEY: [(base_path, base_receipt)]},
        checked_at="2026-07-30T00:00:00Z",
    )
    build_dir = workspace / "build"
    _build(build_dir, 1, previous)
    input_path = workspace / "input.json"
    input_path.write_text("{}\n", encoding="utf-8")

    result = nightly.finalize_nightly(
        feed=feed,
        summary={"catalogs": [{"catalog_key": CATALOG_KEY, "changed": True}]},
        previous_dir=previous_dir,
        build_dir=build_dir,
        public_dir=workspace / "public",
        release_dir=workspace / "release",
        next_state_dir=workspace / "next-state",
        output_feed=workspace / "feed.json",
        tag="catalog-v2-2026-07-31",
        published_at="2026-07-31T00:00:00Z",
        repository="owner/repository",
        commit="a" * 40,
        inputs={"test_input": input_path},
        checked_at="2026-07-31T00:00:00Z",
    )

    assert result["changed"] is True
    assert result["catalogs"][CATALOG_KEY]["version"] == 1
    assert (workspace / "release/verification-audit.json").is_file()
    advanced = load_catalog_feed(workspace / "feed.json")
    entry = advanced.families["milo1"].catalogs["scryfall/mtg"]
    assert entry.current_version == 1
    assert list(entry.updates) == [1]
    state = load_catalog_build(
        workspace / "next-state" / manifest_filename_for_catalog(CATALOG_KEY),
        asset_dir=workspace / "next-state",
    )
    assert state.manifest.version == "1"


def test_finalize_nightly_no_change_creates_no_publication(workspace: Path) -> None:
    result = nightly.finalize_nightly(
        feed=update_catalog_feed,
        summary={"catalogs": []},
        previous_dir=workspace / "previous",
        build_dir=workspace / "build",
        public_dir=workspace / "public",
        release_dir=workspace / "release",
        next_state_dir=workspace / "next-state",
        output_feed=workspace / "feed.json",
        tag="catalog-v2-2026-07-31",
        published_at="2026-07-31T00:00:00Z",
        repository="owner/repository",
        commit="a" * 40,
        inputs={},
        checked_at="2026-07-31T00:00:00Z",
    )

    assert result == {"changed": False, "tag": None, "catalogs": {}}
    assert not (workspace / "release").exists()
