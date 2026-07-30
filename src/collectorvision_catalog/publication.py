from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any

from .artifacts import (
    AssetInfo,
    CatalogBuild,
    CatalogDescriptor,
    SourceRevision,
    ValidationError,
    _require_mapping,
    _require_non_empty_string,
    _requires_delta_upsert,
)
from .versioning import CatalogVersionPlan, validate_public_name, version_root

BASE_ASSETS = ("embeddings", "identifiers", "metadata")
DELTA_ASSETS = {
    "embeddings_delta": "embeddings",
    "identifiers_delta": "identifiers",
    "metadata_delta": "metadata",
}


@dataclass(frozen=True)
class PublishedAsset:
    path: str
    rows: int
    size: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "rows": self.rows,
            "size": self.size,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PublishedAsset:
        path = _relative_asset_path(payload.get("path"), "asset path")
        rows = _non_negative_int(payload.get("rows"), f"asset {path!r} rows")
        size = payload.get("size")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValidationError(f"asset {path!r} size must be a non-negative integer")
        return cls(
            path=path,
            rows=rows,
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
        return cls(
            added=_non_negative_int(payload.get("added"), f"{name}.added"),
            updated=_non_negative_int(payload.get("updated"), f"{name}.updated"),
            deleted=_non_negative_int(payload.get("deleted"), f"{name}.deleted"),
        )


@dataclass(frozen=True)
class DeltaRoute:
    from_version: int
    rows: int
    recognition: ChangeCounts
    metadata: ChangeCounts
    assets: dict[str, PublishedAsset]

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_version": self.from_version,
            "rows": self.rows,
            "recognition": self.recognition.to_dict(),
            "metadata": self.metadata.to_dict(),
            "assets": _assets_to_dict(self.assets),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DeltaRoute:
        from_version = _non_negative_int(payload.get("from_version"), "delta.from_version")
        rows = _positive_int(payload.get("rows"), "delta.rows")
        recognition = ChangeCounts.from_dict(
            _require_mapping(payload.get("recognition"), "delta.recognition"),
            "delta.recognition",
        )
        metadata = ChangeCounts.from_dict(
            _require_mapping(payload.get("metadata"), "delta.metadata"),
            "delta.metadata",
        )
        assets = _assets_from_dict(payload.get("assets"), "delta.assets")
        if recognition.total and "identifiers" not in assets:
            raise ValidationError("recognition delta operations require identifiers")
        if "embeddings" in assets and "identifiers" not in assets:
            raise ValidationError("delta embeddings require identifiers")
        if not recognition.total and {"embeddings", "identifiers"}.intersection(assets):
            raise ValidationError("empty recognition delta cannot contain recognition assets")
        if metadata.total and "metadata" not in assets:
            raise ValidationError("metadata delta operations require metadata")
        if not metadata.total and "metadata" in assets:
            raise ValidationError("empty metadata delta cannot contain metadata")
        if not max(recognition.total, metadata.total) <= rows <= (
            recognition.total + metadata.total
        ):
            raise ValidationError("delta.rows must count unique affected rows")
        if recognition.total and assets["identifiers"].rows != recognition.total:
            raise ValidationError("identifier delta rows do not match recognition changes")
        recognition_upserts = recognition.added + recognition.updated
        if recognition_upserts:
            if "embeddings" not in assets or assets["embeddings"].rows != recognition_upserts:
                raise ValidationError("embedding delta rows do not match recognition upserts")
        elif "embeddings" in assets:
            raise ValidationError("delete-only recognition delta cannot contain embeddings")
        if metadata.total and assets["metadata"].rows != metadata.total:
            raise ValidationError("metadata delta rows do not match metadata changes")
        expected_parent = PurePosixPath(f"delta-from-{from_version}")
        for name, asset in assets.items():
            if name not in BASE_ASSETS:
                raise ValidationError(f"unsupported delta asset {name!r}")
            if PurePosixPath(asset.path).parent != expected_parent:
                raise ValidationError(
                    f"delta asset {name!r} must be under delta-from-{from_version}/"
                )
        return cls(
            from_version=from_version,
            rows=rows,
            recognition=recognition,
            metadata=metadata,
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
    base: dict[str, PublishedAsset] | None
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
            "base": None if self.base is None else _assets_to_dict(self.base),
            "delta": None if self.delta is None else self.delta.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CatalogVersionManifest:
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
            else _assets_from_dict(base_payload, "base")
        )
        delta = (
            None
            if delta_payload is None
            else DeltaRoute.from_dict(_require_mapping(delta_payload, "delta"))
        )
        _validate_routes(version, previous_version, base, delta)
        if base is not None:
            _validate_base_assets(base)
        rows = _positive_int(payload.get("rows"), "rows")
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
    changes = (
        _change_summary(build, previous_build)
        if plan.publish_delta
        else None
    )

    source_root = Path(build_dir)
    version_dir = Path(output_root) / version_root(public_name, plan.version)
    if version_dir.exists():
        raise ValidationError(f"catalog version already exists: {version_dir}")
    version_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = version_dir.with_name(f".{version_dir.name}.tmp")
    if staging_dir.exists():
        raise ValidationError(f"catalog version staging path already exists: {staging_dir}")
    try:
        base_assets = (
            _publish_assets(
                build.manifest.assets,
                BASE_ASSETS,
                source_root,
                staging_dir,
                "base",
                {
                    "embeddings": build.manifest.rows,
                    "identifiers": build.manifest.rows,
                    "metadata": build.manifest.rows,
                },
            )
            if plan.publish_base
            else None
        )
        delta_assets = (
            _publish_assets(
                build.manifest.assets,
                DELTA_ASSETS,
                source_root,
                staging_dir,
                f"delta-from-{plan.previous_version}",
                {
                    "embeddings_delta": changes.recognition.added
                    + changes.recognition.updated,
                    "identifiers_delta": changes.recognition.total,
                    "metadata_delta": changes.metadata.total,
                },
            )
            if plan.publish_delta
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
            base=base_assets,
            delta=(
                None
                if delta_assets is None
                else DeltaRoute(
                    from_version=plan.previous_version,
                    rows=changes.rows,
                    recognition=changes.recognition,
                    metadata=changes.metadata,
                    assets=delta_assets,
                )
            ),
        )
        manifest = CatalogVersionManifest.from_dict(manifest.to_dict())
        (staging_dir / "manifest.json").write_text(
            json.dumps(manifest.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )
        staging_dir.replace(version_dir)
    except BaseException:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise
    return manifest, version_dir / "manifest.json"


def load_catalog_version_manifest(path: str | Path) -> CatalogVersionManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return CatalogVersionManifest.from_dict(_require_mapping(payload, "catalog version manifest"))


def _publish_assets(
    source_assets: Mapping[str, AssetInfo],
    names: tuple[str, ...] | Mapping[str, str],
    source_root: Path,
    version_dir: Path,
    route: str,
    rows_by_asset: Mapping[str, int],
) -> dict[str, PublishedAsset]:
    name_map = {name: name for name in names} if isinstance(names, tuple) else names
    published: dict[str, PublishedAsset] = {}
    for source_name, public_name in name_map.items():
        asset = source_assets.get(source_name)
        if asset is None:
            if source_name in BASE_ASSETS:
                raise ValidationError(f"builder manifest is missing required asset {source_name!r}")
            continue
        filename = _public_filename(public_name, asset.filename)
        relative_path = f"{route}/{filename}"
        destination = version_dir / relative_path
        source = source_root / asset.filename
        payload = source.read_bytes()
        if len(payload) != asset.size or sha256(payload).hexdigest() != asset.sha256:
            raise ValidationError(f"builder asset {source_name!r} failed integrity validation")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        published[public_name] = PublishedAsset(
            path=relative_path,
            rows=rows_by_asset[source_name],
            size=asset.size,
            sha256=asset.sha256,
        )
    return published


def _public_filename(name: str, source_filename: str) -> str:
    suffixes = {
        "embeddings": ".f16.gz",
        "identifiers": ".jsonl.gz",
        "metadata": ".jsonl.gz",
    }
    suffix = suffixes[name]
    if not source_filename.endswith(suffix):
        raise ValidationError(f"unexpected builder filename for {name!r}: {source_filename!r}")
    return f"{name}{suffix}"


def _validate_routes(
    version: int,
    previous_version: int | None,
    base: dict[str, PublishedAsset] | None,
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


def _validate_base_assets(assets: Mapping[str, PublishedAsset]) -> None:
    if set(assets) != set(BASE_ASSETS):
        raise ValidationError(f"base assets must be exactly {list(BASE_ASSETS)}")
    for name, asset in assets.items():
        if PurePosixPath(asset.path).parent != PurePosixPath("base"):
            raise ValidationError(f"base asset {name!r} must be under base/")
    if assets["embeddings"].rows != assets["identifiers"].rows:
        raise ValidationError("base embeddings and identifiers must have equal rows")
    if assets["metadata"].rows != assets["identifiers"].rows:
        raise ValidationError("base metadata and identifiers must have equal rows")


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


@dataclass(frozen=True)
class _ChangeSummary:
    rows: int
    recognition: ChangeCounts
    metadata: ChangeCounts


def _change_summary(build: CatalogBuild, previous_build: CatalogBuild | None) -> _ChangeSummary:
    if previous_build is None:
        raise ValidationError("previous_build is required to publish a delta")
    if build.manifest.previous_version != previous_build.manifest.version:
        raise ValidationError("previous_build version does not match builder manifest")
    if build.manifest.catalog_key != previous_build.manifest.catalog_key:
        raise ValidationError("previous_build catalog does not match builder manifest")
    if build.manifest.embedding_model != previous_build.manifest.embedding_model:
        raise ValidationError("previous_build embedding model does not match builder manifest")
    previous_rows = {row.key: row for row in previous_build.rows}
    current_rows = {row.key: row for row in build.rows}
    previous_keys = set(previous_rows)
    current_keys = set(current_rows)
    recognition_added = current_keys - previous_keys
    recognition_deleted = previous_keys - current_keys
    recognition_updated = {
        key
        for key in previous_keys & current_keys
        if _requires_delta_upsert(previous_rows[key], current_rows[key])
    }

    previous_metadata = {
        key: row.metadata for key, row in previous_rows.items() if row.metadata is not None
    }
    current_metadata = {
        key: row.metadata for key, row in current_rows.items() if row.metadata is not None
    }
    previous_metadata_keys = set(previous_metadata)
    current_metadata_keys = set(current_metadata)
    metadata_added = current_metadata_keys - previous_metadata_keys
    metadata_deleted = previous_metadata_keys - current_metadata_keys
    metadata_updated = {
        key
        for key in previous_metadata_keys & current_metadata_keys
        if previous_metadata[key] != current_metadata[key]
    }
    affected = (
        recognition_added
        | recognition_updated
        | recognition_deleted
        | metadata_added
        | metadata_updated
        | metadata_deleted
    )
    if not affected:
        raise ValidationError("cannot publish an empty delta")
    summary = _ChangeSummary(
        rows=len(affected),
        recognition=ChangeCounts(
            added=len(recognition_added),
            updated=len(recognition_updated),
            deleted=len(recognition_deleted),
        ),
        metadata=ChangeCounts(
            added=len(metadata_added),
            updated=len(metadata_updated),
            deleted=len(metadata_deleted),
        ),
    )
    if summary.recognition.total != build.manifest.delta.operations:
        raise ValidationError("recognition change counts do not match builder delta")
    if summary.metadata.total != build.manifest.delta.metadata_operations:
        raise ValidationError("metadata change counts do not match builder delta")
    return summary
