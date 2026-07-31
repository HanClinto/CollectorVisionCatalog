from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
from conftest import TrackingEmbedder, TrackingImageLoader, build_test_catalog, make_row

from collectorvision_catalog import (
    AUDIT_FILENAME,
    ReleaseAudit,
    ValidationError,
    assemble_catalog_release,
    plan_catalog_version,
    publish_catalog_version,
    validate_catalog_release,
)


def _publication(workspace: Path, catalog_key: str, public_name: str):
    build_dir = workspace / f"build-{public_name}"
    build = build_test_catalog(
        [make_row(public_name, f"memory://{public_name}", f"fp-{public_name}")],
        embedder=TrackingEmbedder(),
        image_loader=TrackingImageLoader({f"memory://{public_name}": (255, 0, 0)}),
        output_dir=build_dir,
        catalog_key=catalog_key,
        version="0",
        embedding_model="milo1",
    )
    receipt, version_dir = publish_catalog_version(
        build,
        build_dir,
        workspace / "public",
        public_name,
        plan_catalog_version(None),
    )
    return version_dir, receipt


def test_release_audit_replaces_manifests_and_checksums(workspace: Path) -> None:
    publications = {
        "milo1/scryfall/mtg": _publication(
            workspace, "milo1/scryfall/mtg", "scryfall-mtg"
        ),
        "milo1/tcgplayer/pokemon": _publication(
            workspace, "milo1/tcgplayer/pokemon", "tcgplayer-pokemon"
        ),
    }
    config = workspace / "catalogs.json"
    config.write_text("{}")
    quality = workspace / "quality.json"
    quality.write_text("{}")
    output = workspace / "release"

    audit = assemble_catalog_release(
        publications,
        output,
        tag="catalog-v2-2026-07-29",
        published_at="2026-07-29T23:00:00Z",
        repository="HanClinto/CollectorVisionCatalog",
        commit="0123456789abcdef",
        inputs={"catalog_config": config, "quality_overrides": quality},
    )

    assert validate_catalog_release(output) == audit
    assert (output / AUDIT_FILENAME).is_file()
    assert not (output / "SHA256SUMS").exists()
    assert not list(output.glob("*.manifest.json"))
    assert set(audit.payload["families"]) == {"milo1"}
    assert set(audit.payload["families"]["milo1"]["catalogs"]) == {
        "scryfall/mtg",
        "tcgplayer/pokemon",
    }
    assert audit.payload["inputs"]["catalog_config"]["sha256"]


def test_release_audit_detects_asset_tampering(workspace: Path) -> None:
    publications = {
        "milo1/scryfall/mtg": _publication(
            workspace, "milo1/scryfall/mtg", "scryfall-mtg"
        )
    }
    config = workspace / "catalogs.json"
    config.write_text("{}")
    output = workspace / "release"
    audit = assemble_catalog_release(
        publications,
        output,
        tag="catalog-v2-2026-07-29",
        published_at="2026-07-29T23:00:00Z",
        repository="HanClinto/CollectorVisionCatalog",
        commit="0123456789abcdef",
        inputs={"catalog_config": config},
    )
    catalog = audit.payload["families"]["milo1"]["catalogs"]["scryfall/mtg"]
    filename = catalog["base"]["assets"]["records"]["filename"]
    (output / filename).write_bytes(b"tampered")

    with pytest.raises(ValidationError, match="failed integrity"):
        validate_catalog_release(output)


def test_release_audit_rejects_malformed_nested_records(workspace: Path) -> None:
    publications = {
        "milo1/scryfall/mtg": _publication(
            workspace, "milo1/scryfall/mtg", "scryfall-mtg"
        )
    }
    config = workspace / "catalogs.json"
    config.write_text("{}")
    audit = assemble_catalog_release(
        publications,
        workspace / "release",
        tag="catalog-v2-2026-07-29",
        published_at="2026-07-29T23:00:00Z",
        repository="HanClinto/CollectorVisionCatalog",
        commit="0123456789abcdef",
        inputs={"catalog_config": config},
    )

    extra_field = deepcopy(audit.to_dict())
    asset = extra_field["families"]["milo1"]["catalogs"]["scryfall/mtg"]["base"][
        "assets"
    ]["records"]
    asset["unexpected"] = True
    with pytest.raises(ValidationError, match="fields must be exactly"):
        ReleaseAudit.from_dict(extra_field)

    unsafe_filename = deepcopy(audit.to_dict())
    asset = unsafe_filename["families"]["milo1"]["catalogs"]["scryfall/mtg"]["base"][
        "assets"
    ]["records"]
    asset["filename"] = "../records.jsonl.gz"
    with pytest.raises(ValidationError, match="safe flat asset filename"):
        ReleaseAudit.from_dict(unsafe_filename)

    noncanonical_timestamp = deepcopy(audit.to_dict())
    noncanonical_timestamp["release"]["published_at"] = "2026-07-29T19:00:00-04:00"
    with pytest.raises(ValidationError, match="normalized RFC3339 UTC"):
        ReleaseAudit.from_dict(noncanonical_timestamp)


def test_release_audit_rejects_missing_inputs_and_mismatched_keys(workspace: Path) -> None:
    publication = _publication(workspace, "milo1/scryfall/mtg", "scryfall-mtg")

    with pytest.raises(ValidationError, match="does not match receipt catalog_key"):
        assemble_catalog_release(
            {"milo1/scryfall/other": publication},
            workspace / "wrong-key-release",
            tag="catalog-v2-2026-07-29",
            published_at="2026-07-29T23:00:00Z",
            repository="HanClinto/CollectorVisionCatalog",
            commit="0123456789abcdef",
            inputs={"catalog_config": workspace / "missing.json"},
        )

    with pytest.raises(ValidationError, match="cannot read audit input"):
        assemble_catalog_release(
            {"milo1/scryfall/mtg": publication},
            workspace / "missing-input-release",
            tag="catalog-v2-2026-07-29",
            published_at="2026-07-29T23:00:00Z",
            repository="HanClinto/CollectorVisionCatalog",
            commit="0123456789abcdef",
            inputs={"catalog_config": workspace / "missing.json"},
        )
