from __future__ import annotations

import gzip
import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest
from conftest import (
    BadEmbedder,
    TrackingEmbedder,
    TrackingImageLoader,
    UnnormalizedEmbedder,
    make_row,
)
from conftest import (
    build_test_catalog as build_catalog,
)

from collectorvision_catalog import (
    AssetIntegrityError,
    CatalogBuild,
    CatalogDescriptor,
    RecognitionRow,
    ValidationError,
    apply_delta,
    load_catalog_build,
    load_delta_bundle,
    manifest_filename_for_catalog,
    validate_artifacts,
)
from collectorvision_catalog.artifacts import AssetInfo, CatalogManifest

CATALOG_KEY = "milo1/scryfall/mtg"
EMBEDDING_MODEL = "milo1"


@pytest.mark.parametrize(
    "filename", ["nested/asset.gz", "../asset.gz", r"nested\asset.gz", "/asset"]
)
def test_asset_info_rejects_non_flat_filenames(filename: str) -> None:
    with pytest.raises(ValidationError, match="flat filename"):
        AssetInfo.from_dict(
            {
                "filename": filename,
                "size": 1,
                "sha256": "abc",
                "content_type": "application/octet-stream",
            }
        )


def _build_initial_catalog(
    workspace: Path,
    rows: list[RecognitionRow],
    image_map: dict[str, tuple[int, int, int]],
) -> tuple[CatalogBuild, Path, TrackingEmbedder, TrackingImageLoader]:
    embedder = TrackingEmbedder()
    loader = TrackingImageLoader(image_map)
    build_dir = workspace / "initial"
    build = build_catalog(
        rows,
        embedder=embedder,
        image_loader=loader,
        output_dir=build_dir,
        catalog_key=CATALOG_KEY,
        version="v1",
        embedding_model=EMBEDDING_MODEL,
        batch_size=2,
    )
    return build, build_dir, embedder, loader


def _read_gzip_jsonl(path: Path) -> list[dict[str, object]]:
    payload = gzip.decompress(path.read_bytes()).decode("utf-8")
    if not payload:
        return []
    return [json.loads(line) for line in payload.splitlines()]


def _replace_gzip_jsonl_asset(
    manifest_path: Path,
    asset_name: str,
    records: list[dict[str, object]],
) -> None:
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    asset = manifest_payload["assets"][asset_name]
    payload = gzip.compress(
        b"".join(
            json.dumps(record, separators=(",", ":"), sort_keys=True).encode("utf-8") + b"\n"
            for record in records
        ),
        mtime=0,
    )
    (manifest_path.parent / asset["filename"]).write_bytes(payload)
    asset["size"] = len(payload)
    asset["sha256"] = sha256(payload).hexdigest()
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")


def _normalized_embedding_for_color(color: tuple[int, int, int]) -> np.ndarray:
    embedding = np.array(
        [color[0] / 255.0, color[1] / 255.0, color[2] / 255.0, 0.5],
        dtype=np.float32,
    )
    embedding /= np.linalg.norm(embedding)
    return embedding


def test_build_outputs_are_deterministic_and_loadable(workspace: Path) -> None:
    rows = [
        make_row(
            "beta",
            "memory://beta",
            "fp-beta",
            identifiers={"scryfall_oracle": "o2"},
            metadata={"name": "Beta", "tags": ["blue", "rare"]},
        ),
        make_row(
            "alpha",
            "memory://alpha",
            "fp-alpha",
            identifiers={"scryfall_oracle": "o1"},
            metadata={"name": "Alpha"},
        ),
    ]
    image_map = {
        "memory://alpha": (255, 0, 0),
        "memory://beta": (0, 255, 0),
    }
    build_a = build_catalog(
        rows,
        embedder=TrackingEmbedder(),
        image_loader=TrackingImageLoader(image_map),
        output_dir=workspace / "build-a",
        catalog_key=CATALOG_KEY,
        version="v1",
        embedding_model=EMBEDDING_MODEL,
        batch_size=2,
    )
    build_b = build_catalog(
        rows,
        embedder=TrackingEmbedder(),
        image_loader=TrackingImageLoader(image_map),
        output_dir=workspace / "build-b",
        catalog_key=CATALOG_KEY,
        version="v1",
        embedding_model=EMBEDDING_MODEL,
        batch_size=2,
    )

    files_a = sorted(path.name for path in (workspace / "build-a").iterdir())
    files_b = sorted(path.name for path in (workspace / "build-b").iterdir())
    assert files_a == files_b
    assert all("/" not in filename for filename in files_a)
    assert manifest_filename_for_catalog(CATALOG_KEY) in files_a
    for filename in files_a:
        payload = (workspace / "build-a" / filename).read_bytes()
        assert payload == (workspace / "build-b" / filename).read_bytes()
        if filename.endswith(".gz"):
            assert payload[:3] == b"\x1f\x8b\x08"
            assert payload[9] == 255

    manifest_path = workspace / "build-a" / manifest_filename_for_catalog(CATALOG_KEY)
    loaded = load_catalog_build(manifest_path)
    delta = load_delta_bundle(manifest_path)
    recognition_records = _read_gzip_jsonl(
        workspace / "build-a" / build_a.manifest.assets["identifiers"].filename
    )

    assert build_a.manifest.to_dict() == build_b.manifest.to_dict()
    assert loaded.manifest.catalog_key == CATALOG_KEY
    assert loaded.manifest.embedding_model == EMBEDDING_MODEL
    assert [row.key for row in loaded.rows] == ["alpha", "beta"]
    assert loaded.embeddings.dtype == np.float16
    assert loaded.embeddings.shape == (2, 4)
    assert np.allclose(
        loaded.embeddings.astype(np.float32),
        build_a.embeddings.astype(np.float32),
        atol=1e-3,
    )
    assert len(delta.operations) == 0
    assert delta.embeddings.shape == (0, 4)
    assert "identifiers_delta" not in build_a.manifest.assets
    assert "embeddings_delta" not in build_a.manifest.assets
    assert "metadata_delta" not in build_a.manifest.assets
    assert loaded.rows[0].metadata == {"name": "Alpha"}
    assert all(
        "image_url" not in record and "image_fingerprint" not in record
        for record in recognition_records
    )


def test_reuses_previous_embeddings_on_metadata_only_change(workspace: Path) -> None:
    rows_v1 = [
        make_row("alpha", "memory://alpha", "fp-alpha", metadata={"name": "Alpha"}),
        make_row("beta", "memory://beta", "fp-beta", metadata={"name": "Beta"}),
    ]
    image_map = {
        "memory://alpha": (255, 0, 0),
        "memory://beta": (0, 255, 0),
    }
    _, build_dir, _, _ = _build_initial_catalog(workspace, rows_v1, image_map)
    previous = load_catalog_build(build_dir / manifest_filename_for_catalog(CATALOG_KEY))

    embedder = TrackingEmbedder()
    loader = TrackingImageLoader(image_map)
    rows_v2 = [
        make_row(
            "alpha",
            "memory://alpha",
            "fp-alpha",
            metadata={"name": "Alpha", "rarity": "rare"},
        ),
        make_row("beta", "memory://beta", "fp-beta", metadata={"name": "Beta"}),
    ]
    build_catalog(
        rows_v2,
        embedder=embedder,
        image_loader=loader,
        output_dir=workspace / "metadata-only",
        catalog_key=CATALOG_KEY,
        version="v2",
        embedding_model=EMBEDDING_MODEL,
        previous_build=previous,
    )
    delta = load_delta_bundle(
        workspace / "metadata-only" / manifest_filename_for_catalog(CATALOG_KEY)
    )

    assert embedder.calls == []
    assert loader.calls == []
    assert len(delta.operations) == 0
    assert len(delta.metadata_operations) == 1


def test_seed_embeddings_reuse_initial_rows_and_keep_artifacts_valid(workspace: Path) -> None:
    rows = [
        make_row("alpha", "memory://alpha", "fp-alpha", metadata={"name": "Alpha"}),
        make_row("beta", "memory://beta", "fp-beta", metadata={"name": "Beta"}),
    ]
    image_map = {
        "memory://alpha": (255, 0, 0),
        "memory://beta": (0, 255, 0),
    }
    embedder = TrackingEmbedder()
    loader = TrackingImageLoader(image_map)
    build = build_catalog(
        rows,
        embedder=embedder,
        image_loader=loader,
        output_dir=workspace / "seeded",
        catalog_key=CATALOG_KEY,
        version="v1",
        embedding_model=EMBEDDING_MODEL,
        seed_embeddings={"alpha": _normalized_embedding_for_color(image_map["memory://alpha"])},
    )

    manifest_path = workspace / "seeded" / manifest_filename_for_catalog(CATALOG_KEY)
    loaded = validate_artifacts(manifest_path)
    delta = load_delta_bundle(manifest_path)

    assert loader.calls == ["memory://beta"]
    assert embedder.calls == [[(0, 255, 0)]]
    assert build.manifest.previous_version is None
    assert build.manifest.delta.base_version is None
    assert build.manifest.delta.requires_exact_base is False
    assert len(delta.operations) == 0
    assert len(delta.metadata_operations) == 0
    assert delta.embeddings.shape == (0, 4)
    with pytest.raises(ValidationError, match="installed from the full snapshot"):
        apply_delta(None, manifest_path)
    assert np.allclose(
        loaded.embeddings[0].astype(np.float32),
        _normalized_embedding_for_color(image_map["memory://alpha"]),
        atol=1e-3,
    )


def test_legacy_empty_delta_assets_remain_readable(workspace: Path) -> None:
    build = build_catalog(
        [make_row("alpha", "memory://alpha", "fp-alpha")],
        embedder=TrackingEmbedder(),
        image_loader=TrackingImageLoader({"memory://alpha": (255, 0, 0)}),
        output_dir=workspace / "legacy",
        catalog_key=CATALOG_KEY,
        version="v1",
        embedding_model=EMBEDDING_MODEL,
    )
    payload = build.manifest.to_dict()
    placeholder = payload["assets"]["identifiers"]
    payload["assets"].update(
        {
            "identifiers_delta": placeholder,
            "embeddings_delta": placeholder,
            "metadata_delta": placeholder,
        }
    )

    assert CatalogManifest.from_dict(payload).delta.operations == 0


def test_identifiers_serialize_and_result_identifier_is_required(workspace: Path) -> None:
    row = make_row(
        "alpha",
        "memory://alpha",
        "fp-alpha",
        identifiers={"scryfall_card": "card-1", "scryfall_oracle": "oracle-1"},
    )
    descriptor = CatalogDescriptor(
        game="magic-the-gathering",
        source="scryfall",
        profile="printings",
        description="Test printings.",
        result_identifier="scryfall_card",
        recommended=True,
    )
    build = build_catalog(
        [row],
        embedder=TrackingEmbedder(),
        image_loader=TrackingImageLoader({"memory://alpha": (255, 0, 0)}),
        output_dir=workspace / "valid-identifiers",
        catalog_key=CATALOG_KEY,
        version="v1",
        embedding_model=EMBEDDING_MODEL,
        descriptor=descriptor,
    )
    assert build.rows[0].minimal_record()["identifiers"] == {
        "scryfall_card": "card-1",
        "scryfall_oracle": "oracle-1",
        "test": "alpha",
    }
    assert build.manifest.descriptor == descriptor

    with pytest.raises(ValidationError, match="missing result identifier"):
        build_catalog(
            [row],
            embedder=TrackingEmbedder(),
            image_loader=TrackingImageLoader({"memory://alpha": (255, 0, 0)}),
            output_dir=workspace / "invalid-identifiers",
            catalog_key=CATALOG_KEY,
            version="v1",
            embedding_model=EMBEDDING_MODEL,
            descriptor=CatalogDescriptor(
                game="magic-the-gathering",
                source="scryfall",
                profile="cards",
                description="Test cards.",
                result_identifier="missing_identifier",
            ),
        )


def test_loader_enforces_manifest_result_identifier(workspace: Path) -> None:
    rows = [make_row("alpha", "memory://alpha", "fp-alpha")]
    _, build_dir, _, _ = _build_initial_catalog(
        workspace,
        rows,
        {"memory://alpha": (255, 0, 0)},
    )
    manifest_path = build_dir / manifest_filename_for_catalog(CATALOG_KEY)
    recognition_path = build_dir / "milo1--scryfall--mtg.identifiers.jsonl.gz"
    recognition_records = _read_gzip_jsonl(recognition_path)
    recognition_records[0]["identifiers"] = {"other": "alpha"}
    _replace_gzip_jsonl_asset(manifest_path, "identifiers", recognition_records)

    with pytest.raises(ValidationError, match="missing result identifier 'test'"):
        load_catalog_build(manifest_path)


def test_manifest_rejects_conflicting_delta_base_versions(workspace: Path) -> None:
    rows = [make_row("alpha", "memory://alpha", "fp-alpha")]
    _, build_dir, _, _ = _build_initial_catalog(
        workspace,
        rows,
        {"memory://alpha": (255, 0, 0)},
    )
    manifest_path = build_dir / manifest_filename_for_catalog(CATALOG_KEY)
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_payload["previous_version"] = "wrong-base"
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")

    with pytest.raises(ValidationError, match="must match delta.base_version"):
        load_catalog_build(manifest_path)


def test_seed_embeddings_reject_unknown_keys(workspace: Path) -> None:
    rows = [make_row("alpha", "memory://alpha", "fp-alpha")]

    with pytest.raises(ValidationError, match="seed_embeddings contain unknown keys"):
        build_catalog(
            rows,
            embedder=TrackingEmbedder(),
            image_loader=TrackingImageLoader({"memory://alpha": (255, 0, 0)}),
            output_dir=workspace / "unknown-seed",
            catalog_key=CATALOG_KEY,
            version="v1",
            embedding_model=EMBEDDING_MODEL,
            seed_embeddings={"beta": _normalized_embedding_for_color((255, 0, 0))},
        )


@pytest.mark.parametrize(
    ("seed_embeddings", "message"),
    [
        (
            {"alpha": np.array([np.nan, 0.0, 0.0, 0.0], dtype=np.float32)},
            "seed_embeddings\\['alpha'\\] embeddings must contain only finite values",
        ),
        (
            {"alpha": np.ones(4, dtype=np.float32)},
            "seed_embeddings\\['alpha'\\] embeddings must be L2-normalized",
        ),
        (
            {
                "alpha": _normalized_embedding_for_color((255, 0, 0)),
                "beta": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            },
            "seed_embeddings\\['beta'\\] returned dimension 3 but expected 4",
        ),
    ],
)
def test_seed_embeddings_reject_invalid_vectors(
    workspace: Path,
    seed_embeddings: dict[str, np.ndarray],
    message: str,
) -> None:
    rows = [
        make_row("alpha", "memory://alpha", "fp-alpha"),
        make_row("beta", "memory://beta", "fp-beta"),
    ]

    with pytest.raises(ValidationError, match=message):
        build_catalog(
            rows,
            embedder=TrackingEmbedder(),
            image_loader=TrackingImageLoader(
                {"memory://alpha": (255, 0, 0), "memory://beta": (0, 255, 0)}
            ),
            output_dir=workspace / "invalid-seed",
            catalog_key=CATALOG_KEY,
            version="v1",
            embedding_model=EMBEDDING_MODEL,
            seed_embeddings=seed_embeddings,
        )


def test_seed_embeddings_cannot_be_combined_with_previous_build(workspace: Path) -> None:
    rows = [make_row("alpha", "memory://alpha", "fp-alpha", metadata={"name": "Alpha"})]
    image_map = {"memory://alpha": (255, 0, 0)}
    _, build_dir, _, _ = _build_initial_catalog(workspace, rows, image_map)
    previous = load_catalog_build(build_dir / manifest_filename_for_catalog(CATALOG_KEY))

    with pytest.raises(ValidationError, match="seed_embeddings cannot be combined"):
        build_catalog(
            rows,
            embedder=TrackingEmbedder(),
            image_loader=TrackingImageLoader(image_map),
            output_dir=workspace / "seed-plus-previous",
            catalog_key=CATALOG_KEY,
            version="v2",
            embedding_model=EMBEDDING_MODEL,
            seed_embeddings={"alpha": _normalized_embedding_for_color((255, 0, 0))},
            previous_build=previous,
        )


def test_state_only_change_reuses_embedding_and_roundtrips_delta(workspace: Path) -> None:
    rows_v1 = [make_row("alpha", "memory://alpha", "fp-alpha", metadata={"name": "Alpha"})]
    image_map = {"memory://alpha": (255, 0, 0)}
    _, build_dir, _, _ = _build_initial_catalog(workspace, rows_v1, image_map)
    previous = load_catalog_build(build_dir / manifest_filename_for_catalog(CATALOG_KEY))

    embedder = TrackingEmbedder()
    loader = TrackingImageLoader({"memory://alpha-mirror": (255, 0, 0)})
    build_catalog(
        [make_row("alpha", "memory://alpha-mirror", "fp-alpha", metadata={"name": "Alpha"})],
        embedder=embedder,
        image_loader=loader,
        output_dir=workspace / "state-change",
        catalog_key=CATALOG_KEY,
        version="v2",
        embedding_model=EMBEDDING_MODEL,
        previous_build=previous,
    )

    manifest_path = workspace / "state-change" / manifest_filename_for_catalog(CATALOG_KEY)
    reconstructed = apply_delta(previous, manifest_path)
    loaded = load_catalog_build(manifest_path)
    delta_records = _read_gzip_jsonl(
        workspace / "state-change" / loaded.manifest.assets["identifiers_delta"].filename
    )

    assert embedder.calls == []
    assert loader.calls == []
    assert delta_records == [
        {
            "embedding_index": 0,
            "op": "upsert",
            "record": loaded.rows[0].minimal_record(),
            "state": loaded.rows[0].state_record().to_dict(),
        }
    ]
    assert reconstructed.rows == loaded.rows
    assert np.array_equal(reconstructed.embeddings, loaded.embeddings)


def test_reembeds_when_image_fingerprint_changes(workspace: Path) -> None:
    rows_v1 = [make_row("alpha", "memory://alpha", "fp-alpha", metadata={"name": "Alpha"})]
    image_map_v1 = {"memory://alpha": (255, 0, 0)}
    _, build_dir, _, _ = _build_initial_catalog(workspace, rows_v1, image_map_v1)
    previous = load_catalog_build(build_dir / manifest_filename_for_catalog(CATALOG_KEY))

    embedder = TrackingEmbedder()
    loader = TrackingImageLoader({"memory://alpha-new": (0, 0, 255)})
    build = build_catalog(
        [make_row("alpha", "memory://alpha-new", "fp-alpha-new", metadata={"name": "Alpha"})],
        embedder=embedder,
        image_loader=loader,
        output_dir=workspace / "image-change",
        catalog_key=CATALOG_KEY,
        version="v2",
        embedding_model=EMBEDDING_MODEL,
        previous_build=previous,
    )
    delta = load_delta_bundle(
        workspace / "image-change" / manifest_filename_for_catalog(CATALOG_KEY)
    )

    assert len(embedder.calls) == 1
    assert loader.calls == ["memory://alpha-new"]
    assert len(delta.operations) == 1
    assert not np.allclose(
        previous.embeddings.astype(np.float32),
        build.embeddings.astype(np.float32),
    )


def test_apply_delta_roundtrip_with_add_delete_and_minimal_change(workspace: Path) -> None:
    rows_v1 = [
        make_row("alpha", "memory://alpha", "fp-alpha", metadata={"name": "Alpha"}),
        make_row(
            "beta",
            "memory://beta",
            "fp-beta",
            identifiers={"scryfall_oracle": "old"},
            metadata={"name": "Beta", "rarity": "common"},
        ),
    ]
    image_map_v1 = {
        "memory://alpha": (255, 0, 0),
        "memory://beta": (0, 255, 0),
    }
    _, build_dir, _, _ = _build_initial_catalog(workspace, rows_v1, image_map_v1)
    previous = load_catalog_build(build_dir / manifest_filename_for_catalog(CATALOG_KEY))

    rows_v2 = [
        make_row(
            "beta",
            "memory://beta",
            "fp-beta",
            identifiers={"scryfall_oracle": "new"},
            metadata={"name": "Beta", "rarity": "rare"},
        ),
        make_row("gamma", "memory://gamma", "fp-gamma", metadata={"name": "Gamma"}),
    ]
    image_map_v2 = {
        "memory://beta": (0, 255, 0),
        "memory://gamma": (0, 0, 255),
    }
    embedder = TrackingEmbedder()
    loader = TrackingImageLoader(image_map_v2)
    build_catalog(
        rows_v2,
        embedder=embedder,
        image_loader=loader,
        output_dir=workspace / "delta-target",
        catalog_key=CATALOG_KEY,
        version="v2",
        embedding_model=EMBEDDING_MODEL,
        previous_build=previous,
    )

    target_manifest = workspace / "delta-target" / manifest_filename_for_catalog(CATALOG_KEY)
    delta = load_delta_bundle(target_manifest)
    reconstructed = apply_delta(previous, target_manifest)
    loaded_target = validate_artifacts(target_manifest, previous_build=previous)

    assert loader.calls == ["memory://gamma"]
    assert len(delta.operations) == 3
    assert (
        [row.key for row in reconstructed.rows]
        == [row.key for row in loaded_target.rows]
        == [
            "beta",
            "gamma",
        ]
    )
    assert [row.minimal_record() for row in reconstructed.rows] == [
        row.minimal_record() for row in loaded_target.rows
    ]
    assert [row.state_record() for row in reconstructed.rows] == [
        row.state_record() for row in loaded_target.rows
    ]
    assert [row.metadata for row in reconstructed.rows] == [
        row.metadata for row in loaded_target.rows
    ]
    assert np.allclose(
        reconstructed.embeddings.astype(np.float32),
        loaded_target.embeddings.astype(np.float32),
        atol=1e-3,
    )


def test_loader_rejects_corruption_and_truncation(workspace: Path) -> None:
    rows = [make_row("alpha", "memory://alpha", "fp-alpha", metadata={"name": "Alpha"})]
    image_map = {"memory://alpha": (255, 0, 0)}
    _, build_dir, _, _ = _build_initial_catalog(workspace, rows, image_map)
    manifest_path = build_dir / manifest_filename_for_catalog(CATALOG_KEY)

    recognition_rows_path = build_dir / "milo1--scryfall--mtg.identifiers.jsonl.gz"
    recognition_rows_path.write_bytes(recognition_rows_path.read_bytes() + b"corrupt")
    with pytest.raises(AssetIntegrityError):
        load_catalog_build(manifest_path)

    _, build_dir, _, _ = _build_initial_catalog(workspace / "truncate", rows, image_map)
    manifest_path = build_dir / manifest_filename_for_catalog(CATALOG_KEY)
    truncated_asset_path = build_dir / "milo1--scryfall--mtg.embeddings.f16.gz"
    truncated_bytes = truncated_asset_path.read_bytes()[:-3]
    truncated_asset_path.write_bytes(truncated_bytes)
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_payload["assets"]["embeddings"]["size"] = len(truncated_bytes)
    manifest_payload["assets"]["embeddings"]["sha256"] = sha256(truncated_bytes).hexdigest()
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")
    with pytest.raises(AssetIntegrityError):
        load_catalog_build(manifest_path)


@pytest.mark.parametrize(
    ("rows", "embedder", "message"),
    [
        ([], TrackingEmbedder(), "source rows must not be empty"),
        (
            [
                make_row("dup", "memory://alpha", "fp-a"),
                make_row("dup", "memory://beta", "fp-b"),
            ],
            TrackingEmbedder(),
            "duplicate key 'dup'",
        ),
        (
            [make_row("alpha", "memory://alpha", "fp-alpha")],
            BadEmbedder(),
            "embeddings must contain only finite values",
        ),
        (
            [make_row("alpha", "memory://alpha", "fp-alpha")],
            UnnormalizedEmbedder(),
            "embeddings must be L2-normalized",
        ),
    ],
)
def test_build_rejects_duplicate_and_invalid_rows(
    workspace: Path,
    rows: list[RecognitionRow],
    embedder: object,
    message: str,
) -> None:
    loader = TrackingImageLoader({"memory://alpha": (255, 0, 0), "memory://beta": (0, 255, 0)})
    with pytest.raises(ValidationError, match=message):
        build_catalog(
            rows,
            embedder=embedder,
            image_loader=loader,
            output_dir=workspace / "invalid",
            catalog_key=CATALOG_KEY,
            version="v1",
            embedding_model=EMBEDDING_MODEL,
        )


def test_build_rejects_embedding_model_mismatch(workspace: Path) -> None:
    rows = [make_row("alpha", "memory://alpha", "fp-alpha", metadata={"name": "Alpha"})]
    image_map = {"memory://alpha": (255, 0, 0)}
    _, build_dir, _, _ = _build_initial_catalog(workspace, rows, image_map)
    previous = load_catalog_build(build_dir / manifest_filename_for_catalog(CATALOG_KEY))

    with pytest.raises(ValidationError, match="previous build embedding_model"):
        build_catalog(
            rows,
            embedder=TrackingEmbedder(),
            image_loader=TrackingImageLoader(image_map),
            output_dir=workspace / "mismatch",
            catalog_key=CATALOG_KEY,
            version="v2",
            embedding_model="other-model",
            previous_build=previous,
        )
