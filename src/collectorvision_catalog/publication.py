from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .artifacts import (
    METADATA_UNSET,
    AssetInfo,
    CatalogBuild,
    CatalogDescriptor,
    PublicDelete,
    PublicOperation,
    PublicUpsert,
    RecognitionRow,
    SourceRevision,
    ValidationError,
    _require_mapping,
    _require_non_empty_string,
    _requires_delta_upsert,
    build_public_base_records,
    build_public_delta_records,
    build_public_embeddings,
)
from .versioning import CatalogVersionPlan, validate_public_name, version_root

BASE_ASSET_NAMES = {"records", "embeddings"}
DELTA_ASSET_NAMES = {"records", "embeddings"}


def _exact_fields(payload: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(payload) != expected:
        raise ValidationError(f"{name} fields must be exactly {sorted(expected)}")


@dataclass(frozen=True)
class PublishedAsset:
    path: str
    size: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "size": self.size,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PublishedAsset:
        _exact_fields(payload, {"path", "size", "sha256"}, "published asset")
        path = _relative_asset_path(payload.get("path"), "asset path")
        size = payload.get("size")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValidationError(f"asset {path!r} size must be a non-negative integer")
        return cls(
            path=path,
            size=size,
            sha256=_require_non_empty_string(payload.get("sha256"), f"asset {path!r} sha256"),
        )


@dataclass(frozen=True)
class ChangeCounts:
    added: int
    updated: int
    deleted: int

    @property
    def total(self) -> int:
        return self.added + self.updated + self.deleted

    def to_dict(self) -> dict[str, int]:
        return {
            "added": self.added,
            "updated": self.updated,
            "deleted": self.deleted,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], name: str) -> ChangeCounts:
        _exact_fields(payload, {"added", "updated", "deleted"}, name)
        return cls(
            added=_non_negative_int(payload.get("added"), f"{name}.added"),
            updated=_non_negative_int(payload.get("updated"), f"{name}.updated"),
            deleted=_non_negative_int(payload.get("deleted"), f"{name}.deleted"),
        )


@dataclass(frozen=True)
class BaseRoute:
    rows: int
    assets: dict[str, PublishedAsset]

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "assets": _assets_to_dict(self.assets),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> BaseRoute:
        _exact_fields(payload, {"rows", "assets"}, "base")
        assets = _assets_from_dict(payload.get("assets"), "base.assets")
        if set(assets) != BASE_ASSET_NAMES:
            raise ValidationError("base assets must be exactly records and embeddings")
        route = cls(
            rows=_positive_int(payload.get("rows"), "base.rows"),
            assets=assets,
        )
        _validate_route_asset_paths(route.assets, PurePosixPath("base"), "base")
        return route


@dataclass(frozen=True)
class DeltaRoute:
    from_version: int
    rows: ChangeCounts
    recognition_rows: int
    metadata_rows: int
    assets: dict[str, PublishedAsset]

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_version": self.from_version,
            "rows": self.rows.to_dict(),
            "recognition_rows": self.recognition_rows,
            "metadata_rows": self.metadata_rows,
            "assets": _assets_to_dict(self.assets),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DeltaRoute:
        _exact_fields(
            payload,
            {"from_version", "rows", "recognition_rows", "metadata_rows", "assets"},
            "delta",
        )
        from_version = _non_negative_int(payload.get("from_version"), "delta.from_version")
        rows = ChangeCounts.from_dict(
            _require_mapping(payload.get("rows"), "delta.rows"), "delta.rows"
        )
        if rows.total == 0:
            raise ValidationError("delta rows must not be empty")
        recognition_rows = _non_negative_int(
            payload.get("recognition_rows"), "delta.recognition_rows"
        )
        metadata_rows = _non_negative_int(payload.get("metadata_rows"), "delta.metadata_rows")
        assets = _assets_from_dict(payload.get("assets"), "delta.assets")
        if not set(assets).issubset(DELTA_ASSET_NAMES):
            raise ValidationError("delta contains unsupported assets")
        if "records" not in assets:
            raise ValidationError("delta must include a records asset")
        if "embeddings" in assets and recognition_rows == 0:
            raise ValidationError("delta embeddings require recognition changes")
        expected_parent = PurePosixPath(f"delta-from-{from_version}")
        _validate_route_asset_paths(assets, expected_parent, "delta")
        if not max(recognition_rows, metadata_rows) <= rows.total <= (
            recognition_rows + metadata_rows
        ):
            raise ValidationError("delta rows must count unique affected catalog rows")
        return cls(
            from_version=from_version,
            rows=rows,
            recognition_rows=recognition_rows,
            metadata_rows=metadata_rows,
            assets=assets,
        )


@dataclass(frozen=True)
class CatalogVersionManifest:
    catalog_key: str
    public_name: str
    version: int
    previous_version: int | None
    embedding_model: str
    source_revision: SourceRevision
    descriptor: CatalogDescriptor
    rows: int
    dim: int
    dtype: str
    base: BaseRoute | None
    delta: DeltaRoute | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "catalog_key": self.catalog_key,
            "public_name": self.public_name,
            "version": self.version,
            "previous_version": self.previous_version,
            "embedding_model": self.embedding_model,
            "source_revision": self.source_revision.to_dict(),
            "descriptor": self.descriptor.to_dict(),
            "rows": self.rows,
            "dim": self.dim,
            "dtype": self.dtype,
            "base": None if self.base is None else self.base.to_dict(),
            "delta": None if self.delta is None else self.delta.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CatalogVersionManifest:
        _exact_fields(
            payload,
            {
                "catalog_key",
                "public_name",
                "version",
                "previous_version",
                "embedding_model",
                "source_revision",
                "descriptor",
                "rows",
                "dim",
                "dtype",
                "base",
                "delta",
            },
            "catalog version manifest",
        )
        version = _non_negative_int(payload.get("version"), "version")
        previous_version_payload = payload.get("previous_version")
        previous_version = (
            None
            if previous_version_payload is None
            else _non_negative_int(previous_version_payload, "previous_version")
        )
        base_payload = payload.get("base")
        delta_payload = payload.get("delta")
        base = (
            None
            if base_payload is None
            else BaseRoute.from_dict(_require_mapping(base_payload, "base"))
        )
        delta = (
            None
            if delta_payload is None
            else DeltaRoute.from_dict(_require_mapping(delta_payload, "delta"))
        )
        _validate_routes(version, previous_version, base, delta)
        rows = _positive_int(payload.get("rows"), "rows")
        if base is not None and base.rows != rows:
            raise ValidationError("base.rows must match manifest rows")
        dim = _positive_int(payload.get("dim"), "dim")
        if payload.get("dtype") != "float16":
            raise ValidationError("dtype must be 'float16'")
        return cls(
            catalog_key=_require_non_empty_string(payload.get("catalog_key"), "catalog_key"),
            public_name=validate_public_name(payload.get("public_name")),
            version=version,
            previous_version=previous_version,
            embedding_model=_require_non_empty_string(
                payload.get("embedding_model"), "embedding_model"
            ),
            source_revision=SourceRevision.from_dict(
                _require_mapping(payload.get("source_revision"), "source_revision")
            ),
            descriptor=CatalogDescriptor.from_dict(
                _require_mapping(payload.get("descriptor"), "descriptor")
            ),
            rows=rows,
            dim=dim,
            dtype="float16",
            base=base,
            delta=delta,
        )


def publish_catalog_version(
    build: CatalogBuild,
    build_dir: str | Path,
    output_root: str | Path,
    public_name: str,
    plan: CatalogVersionPlan,
    *,
    previous_build: CatalogBuild | None = None,
) -> tuple[CatalogVersionManifest, Path]:
    public_name = validate_public_name(public_name)
    if _builder_version(build.manifest.version, "version") != plan.version:
        raise ValidationError("builder manifest version does not match publication plan")
    expected_previous = (
        None
        if build.manifest.previous_version is None
        else _builder_version(build.manifest.previous_version, "previous_version")
    )
    if expected_previous != plan.previous_version:
        raise ValidationError("builder previous_version does not match publication plan")

    source_root = Path(build_dir)
    _verify_builder_assets(build.manifest.assets, source_root)

    delta_summary = (
        _build_public_delta(build, previous_build) if plan.publish_delta else None
    )

    version_dir = Path(output_root) / version_root(public_name, plan.version)
    if version_dir.exists():
        raise ValidationError(f"catalog version already exists: {version_dir}")
    version_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = version_dir.with_name(f".{version_dir.name}.tmp")
    if staging_dir.exists():
        raise ValidationError(f"catalog version staging path already exists: {staging_dir}")
    try:
        base_route = _publish_base(build, staging_dir) if plan.publish_base else None
        delta_route = (
            _publish_delta_route(delta_summary, plan.previous_version, staging_dir)
            if delta_summary is not None
            else None
        )
        manifest = CatalogVersionManifest(
            catalog_key=build.manifest.catalog_key,
            public_name=public_name,
            version=plan.version,
            previous_version=plan.previous_version,
            embedding_model=build.manifest.embedding_model,
            source_revision=build.manifest.source_revision,
            descriptor=build.manifest.descriptor,
            rows=build.manifest.rows,
            dim=build.manifest.dim,
            dtype=build.manifest.dtype,
            base=base_route,
            delta=delta_route,
        )
        manifest = CatalogVersionManifest.from_dict(manifest.to_dict())
        staging_dir.replace(version_dir)
    except BaseException:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise
    return manifest, version_dir


def load_catalog_version_manifest(path: str | Path) -> CatalogVersionManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return CatalogVersionManifest.from_dict(_require_mapping(payload, "catalog version manifest"))


def _verify_builder_assets(assets: Mapping[str, AssetInfo], source_root: Path) -> None:
    """Re-verify every builder asset on disk still matches its own manifest.

    Publication rebuilds public payloads from the in-memory ``CatalogBuild``
    rather than re-reading builder files, so this defends against a builder
    output directory that was tampered with or corrupted after the build.
    """
    for name, asset in assets.items():
        path = source_root / asset.filename
        try:
            payload = path.read_bytes()
        except OSError as error:
            raise ValidationError(f"cannot read builder asset {name!r}: {error}") from error
        if len(payload) != asset.size or sha256(payload).hexdigest() != asset.sha256:
            raise ValidationError(f"builder asset {name!r} failed integrity validation")


def _publish_base(build: CatalogBuild, staging_dir: Path) -> BaseRoute:
    route_dir = staging_dir / "base"
    records_payload = build_public_base_records(build.rows)
    embeddings_payload = build_public_embeddings(build.embeddings)
    assets = {
        "records": _write_public_asset(records_payload, "records.jsonl.gz", route_dir),
        "embeddings": _write_public_asset(embeddings_payload, "embeddings.f16.gz", route_dir),
    }
    return BaseRoute(rows=build.manifest.rows, assets=assets)


@dataclass(frozen=True)
class _PublicDeltaSummary:
    operations: tuple[PublicOperation, ...]
    embeddings: NDArray[np.float16]
    rows: ChangeCounts
    recognition_rows: int
    metadata_rows: int


def _build_public_delta(
    build: CatalogBuild, previous_build: CatalogBuild | None
) -> _PublicDeltaSummary:
    if previous_build is None:
        raise ValidationError("previous_build is required to publish a delta")
    if build.manifest.previous_version != previous_build.manifest.version:
        raise ValidationError("previous_build version does not match builder manifest")
    if build.manifest.catalog_key != previous_build.manifest.catalog_key:
        raise ValidationError("previous_build catalog does not match builder manifest")
    if build.manifest.embedding_model != previous_build.manifest.embedding_model:
        raise ValidationError(
            "previous_build embedding model does not match builder manifest"
        )

    previous_rows: dict[str, RecognitionRow] = {row.key: row for row in previous_build.rows}
    current_rows: dict[str, RecognitionRow] = {row.key: row for row in build.rows}
    current_index = {row.key: index for index, row in enumerate(build.rows)}
    previous_keys = set(previous_rows)
    current_keys = set(current_rows)
    deleted_keys = previous_keys - current_keys
    added_keys = current_keys - previous_keys
    surviving_keys = previous_keys & current_keys
    recognition_changed = {
        key
        for key in surviving_keys
        if _requires_delta_upsert(previous_rows[key], current_rows[key])
    } | added_keys

    previous_metadata = {
        key: row.metadata for key, row in previous_rows.items() if row.metadata is not None
    }
    current_metadata = {
        key: row.metadata for key, row in current_rows.items() if row.metadata is not None
    }
    metadata_changed = {
        key
        for key in surviving_keys | added_keys
        if previous_metadata.get(key) != current_metadata.get(key)
    }

    if not deleted_keys and not recognition_changed and not metadata_changed:
        raise ValidationError("cannot publish an empty delta")

    operations: list[PublicOperation] = []
    embedding_rows: list[NDArray[np.float16]] = []
    recognition_rows_count = 0
    metadata_rows_count = 0
    for key in sorted(deleted_keys):
        previous_row = previous_rows[key]
        operations.append(PublicDelete(id=previous_row.id, face_index=previous_row.face_index))
        recognition_rows_count += 1
        if key in previous_metadata:
            metadata_rows_count += 1

    for key in sorted(recognition_changed | metadata_changed):
        row = current_rows[key]
        needs_embedding = key in recognition_changed
        metadata_touched = key in metadata_changed
        embedding_index: int | None = None
        if needs_embedding:
            embedding_index = len(embedding_rows)
            embedding_rows.append(build.embeddings[current_index[key]])
            recognition_rows_count += 1
        metadata_value = current_metadata.get(key) if metadata_touched else METADATA_UNSET
        if metadata_touched:
            metadata_rows_count += 1
        operations.append(
            PublicUpsert(
                row=row.with_metadata(None),
                metadata=metadata_value,
                embedding_index=embedding_index,
            )
        )

    rows = ChangeCounts(
        added=len(added_keys),
        updated=len((recognition_changed | metadata_changed) - added_keys),
        deleted=len(deleted_keys),
    )
    embeddings = (
        np.vstack(embedding_rows).astype(np.float16, copy=False)
        if embedding_rows
        else np.empty((0, build.embeddings.shape[1]), dtype=np.float16)
    )
    if recognition_rows_count != build.manifest.delta.operations:
        raise ValidationError("recognition change counts do not match builder delta")
    if metadata_rows_count != build.manifest.delta.metadata_operations:
        raise ValidationError("metadata change counts do not match builder delta")
    return _PublicDeltaSummary(
        operations=tuple(operations),
        embeddings=embeddings,
        rows=rows,
        recognition_rows=recognition_rows_count,
        metadata_rows=metadata_rows_count,
    )


def _publish_delta_route(
    summary: _PublicDeltaSummary, previous_version: int, staging_dir: Path
) -> DeltaRoute:
    route_dir = staging_dir / f"delta-from-{previous_version}"
    assets = {
        "records": _write_public_asset(
            build_public_delta_records(summary.operations), "records.jsonl.gz", route_dir
        ),
    }
    if len(summary.embeddings):
        assets["embeddings"] = _write_public_asset(
            build_public_embeddings(summary.embeddings), "embeddings.f16.gz", route_dir
        )
    return DeltaRoute(
        from_version=previous_version,
        rows=summary.rows,
        recognition_rows=summary.recognition_rows,
        metadata_rows=summary.metadata_rows,
        assets=assets,
    )


def _write_public_asset(payload: bytes, filename: str, directory: Path) -> PublishedAsset:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / filename).write_bytes(payload)
    return PublishedAsset(
        path=f"{directory.name}/{filename}",
        size=len(payload),
        sha256=sha256(payload).hexdigest(),
    )


def _validate_routes(
    version: int,
    previous_version: int | None,
    base: BaseRoute | None,
    delta: DeltaRoute | None,
) -> None:
    if version == 0:
        if previous_version is not None or base is None or delta is not None:
            raise ValidationError("version 0 must contain only a base")
    elif previous_version != version - 1:
        raise ValidationError("catalog versions must advance exactly one version")
    if version > 0 and base is None and delta is None:
        raise ValidationError("catalog version must contain a base or delta")
    if delta is not None and delta.from_version != previous_version:
        raise ValidationError("delta.from_version must match previous_version")


def _validate_route_asset_paths(
    assets: Mapping[str, PublishedAsset],
    expected_parent: PurePosixPath,
    name: str,
) -> None:
    for asset_name, asset in assets.items():
        if PurePosixPath(asset.path).parent != expected_parent:
            raise ValidationError(
                f"{name} asset {asset_name!r} must be under {expected_parent}/"
            )


def _assets_to_dict(assets: Mapping[str, PublishedAsset]) -> dict[str, Any]:
    return {name: asset.to_dict() for name, asset in sorted(assets.items())}


def _assets_from_dict(value: Any, name: str) -> dict[str, PublishedAsset]:
    return {
        _require_non_empty_string(asset_name, f"{name} key"): PublishedAsset.from_dict(
            _require_mapping(asset_payload, f"{name}[{asset_name!r}]")
        )
        for asset_name, asset_payload in _require_mapping(value, name).items()
    }


def _relative_asset_path(value: Any, name: str) -> str:
    path = _require_non_empty_string(value, name)
    parsed = PurePosixPath(path)
    if parsed.is_absolute() or ".." in parsed.parts or "\\" in path or len(parsed.parts) != 2:
        raise ValidationError(f"{name} must be a safe route-relative path")
    return path


def _non_negative_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValidationError(f"{name} must be a non-negative integer")
    return value


def _positive_int(value: Any, name: str) -> int:
    value = _non_negative_int(value, name)
    if value == 0:
        raise ValidationError(f"{name} must be a positive integer")
    return value


def _builder_version(value: str, name: str) -> int:
    if not value.isascii() or not value.isdecimal():
        raise ValidationError(f"builder manifest {name} must be a non-negative integer")
    return int(value)
