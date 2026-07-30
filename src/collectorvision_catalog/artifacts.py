from __future__ import annotations

import gzip
import io
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol
from urllib.request import urlopen

import numpy as np
from numpy.typing import ArrayLike, NDArray
from PIL import Image

JSONValue = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]

_SAFE_SLUG_CHARS = re.compile(r"[^A-Za-z0-9._-]+")
_PRIMARY_IDENTIFIERS = {
    "scryfall": "scryfall_card",
    "tcgplayer": "tcgplayer_product",
}
_REQUIRED_ASSET_NAMES = {
    "embeddings",
    "identifiers",
    "metadata",
    "state_rows",
}


class CatalogError(ValueError):
    """Base error for catalog operations."""


class ValidationError(CatalogError):
    """Raised when source rows or embeddings are invalid."""


class AssetIntegrityError(CatalogError):
    """Raised when an asset fails checksum, size, or decode validation."""


class Embedder(Protocol):
    def __call__(
        self, images: Sequence[Image.Image]
    ) -> Sequence[Sequence[float]] | NDArray[np.floating[Any]]: ...


class ImageLoader(Protocol):
    def __call__(self, image_url: str) -> Image.Image: ...


def normalize_rfc3339_utc(value: str) -> str:
    """Normalize an RFC3339 timestamp to its canonical UTC representation."""
    text = _require_non_empty_string(value, "source updated_at")
    if re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?"
        r"(?:[Zz]|[+-]\d{2}:?\d{2})",
        text,
    ) is None:
        raise ValidationError("source updated_at must be an RFC3339 timestamp")
    candidate = text
    if re.fullmatch(r".*[+-]\d{4}", candidate):
        candidate = f"{candidate[:-2]}:{candidate[-2:]}"
    if candidate.endswith(("Z", "z")):
        candidate = f"{candidate[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as error:
        raise ValidationError("source updated_at must be an RFC3339 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValidationError("source updated_at must include a UTC offset")
    normalized = parsed.astimezone(timezone.utc).isoformat(timespec="microseconds")
    normalized = normalized.removesuffix("+00:00")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    normalized = f"{normalized}Z"
    return normalized


def max_source_updated_at(revisions: Iterable[SourceRevision]) -> str:
    values = [revision.updated_at for revision in revisions]
    if not values:
        raise ValidationError("at least one source revision is required")
    return max(values, key=lambda value: datetime.fromisoformat(value.replace("Z", "+00:00")))


@dataclass(frozen=True)
class SourceRevision:
    source_type: str
    source_name: str
    updated_at: str
    uri: str
    identity: str

    def __post_init__(self) -> None:
        for value, label in (
            (self.source_type, "source type"),
            (self.source_name, "source name"),
            (self.uri, "source uri"),
            (self.identity, "source identity"),
        ):
            if _require_non_empty_string(value, label) != value:
                raise ValidationError(f"{label} must not contain surrounding whitespace")
        if normalize_rfc3339_utc(self.updated_at) != self.updated_at:
            raise ValidationError("source updated_at must be normalized RFC3339 UTC")

    def to_dict(self) -> dict[str, str]:
        return {
            "type": self.source_type,
            "name": self.source_name,
            "updated_at": self.updated_at,
            "uri": self.uri,
            "identity": self.identity,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SourceRevision:
        expected = {"type", "name", "updated_at", "uri", "identity"}
        if set(payload) != expected:
            raise ValidationError(
                f"source revision fields must be exactly {sorted(expected)}"
            )
        return cls(
            source_type=_require_non_empty_string(payload.get("type"), "source type"),
            source_name=_require_non_empty_string(payload.get("name"), "source name"),
            updated_at=_require_non_empty_string(
                payload.get("updated_at"), "source updated_at"
            ),
            uri=_require_non_empty_string(payload.get("uri"), "source uri"),
            identity=_require_non_empty_string(payload.get("identity"), "source identity"),
        )


@dataclass(frozen=True)
class CatalogDescriptor:
    game: str
    source: str
    profile: str
    description: str
    result_identifier: str
    recommended: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "game": self.game,
            "source": self.source,
            "profile": self.profile,
            "description": self.description,
            "result_identifier": self.result_identifier,
            "recommended": self.recommended,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CatalogDescriptor:
        recommended = payload.get("recommended", False)
        if not isinstance(recommended, bool):
            raise ValidationError("descriptor.recommended must be a boolean")
        return cls(
            game=_require_non_empty_string(payload.get("game"), "descriptor.game"),
            source=_require_non_empty_string(payload.get("source"), "descriptor.source"),
            profile=_require_non_empty_string(payload.get("profile"), "descriptor.profile"),
            description=_require_non_empty_string(
                payload.get("description"), "descriptor.description"
            ),
            result_identifier=_require_non_empty_string(
                payload.get("result_identifier"), "descriptor.result_identifier"
            ),
            recommended=recommended,
        )


@dataclass(frozen=True)
class StateRecord:
    key: str
    image_url: str
    image_fingerprint: str

    def to_dict(self) -> dict[str, str]:
        return {
            "image_url": self.image_url,
            "image_fingerprint": self.image_fingerprint,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], *, key: str) -> StateRecord:
        _require_exact_fields(payload, {"image_url", "image_fingerprint"}, "state")
        return cls(
            key=key,
            image_url=_require_non_empty_string(payload.get("image_url"), "state.image_url"),
            image_fingerprint=_require_non_empty_string(
                payload.get("image_fingerprint"),
                "state.image_fingerprint",
            ),
        )


@dataclass(frozen=True)
class RecognitionRow:
    provider: str
    id: str
    identifiers: dict[str, str]
    image_url: str
    image_fingerprint: str
    face_index: int = 0
    metadata: dict[str, JSONValue] | None = None

    @property
    def key(self) -> str:
        return catalog_row_key(self.provider, self.id, self.face_index)

    def minimal_record(self) -> dict[str, Any]:
        record = {
            "id": self.id,
            "identifiers": dict(sorted(self.identifiers.items())),
        }
        if self.face_index:
            record["face_index"] = self.face_index
        return record

    def state_record(self) -> StateRecord:
        return StateRecord(
            key=self.key,
            image_url=self.image_url,
            image_fingerprint=self.image_fingerprint,
        )

    def with_metadata(self, metadata: dict[str, JSONValue] | None) -> RecognitionRow:
        return RecognitionRow(
            provider=self.provider,
            id=self.id,
            identifiers=dict(self.identifiers),
            image_url=self.image_url,
            image_fingerprint=self.image_fingerprint,
            face_index=self.face_index,
            metadata=metadata,
        )

    @classmethod
    def from_artifact_records(
        cls,
        minimal_payload: Mapping[str, Any],
        state_payload: Mapping[str, Any] | StateRecord,
        *,
        provider: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> RecognitionRow:
        allowed_fields = {"id", "identifiers", "face_index"}
        _require_allowed_fields(
            minimal_payload,
            required={"id", "identifiers"},
            allowed=allowed_fields,
            name="recognition record",
        )
        provider = _identity_component(provider, "provider")
        primary_id = _identity_component(minimal_payload.get("id"), "id")
        identifiers = {
            _require_non_empty_string(namespace, "identifiers key"): _require_non_empty_string(
                value,
                f"identifiers[{namespace!r}]",
            )
            for namespace, value in _require_mapping(
                minimal_payload.get("identifiers"),
                "identifiers",
            ).items()
        }
        face_index = minimal_payload.get("face_index", 0)
        if (
            not isinstance(face_index, int)
            or isinstance(face_index, bool)
            or face_index < 0
        ):
            raise ValidationError("face_index must be a non-negative integer")
        if face_index == 0 and "face_index" in minimal_payload:
            raise ValidationError("recognition record must omit face_index for face 0")
        key = catalog_row_key(provider, primary_id, face_index)
        state = state_payload if isinstance(state_payload, StateRecord) else StateRecord.from_dict(
            state_payload, key=key
        )
        if state.key != key:
            raise ValidationError(
                f"state entry key {state.key!r} does not match recognition key {key!r}"
            )
        normalized_metadata = None if metadata is None else _canonicalize_metadata(metadata)
        return cls(
            provider=provider,
            id=primary_id,
            identifiers=dict(sorted(identifiers.items())),
            image_url=state.image_url,
            image_fingerprint=state.image_fingerprint,
            face_index=face_index,
            metadata=normalized_metadata,
        )


@dataclass(frozen=True)
class AssetInfo:
    filename: str
    size: int
    sha256: str
    content_encoding: str | None
    content_type: str

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "filename": self.filename,
            "size": self.size,
            "sha256": self.sha256,
            "content_type": self.content_type,
        }
        if self.content_encoding is not None:
            payload["content_encoding"] = self.content_encoding
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AssetInfo:
        filename = _require_non_empty_string(payload.get("filename"), "asset filename")
        if (
            filename in {".", ".."}
            or "/" in filename
            or "\\" in filename
            or Path(filename).is_absolute()
        ):
            raise ValidationError("asset filename must be a flat filename")
        size = payload.get("size")
        if not isinstance(size, int) or size < 0:
            raise ValidationError(f"asset {filename!r} size must be a non-negative integer")
        return cls(
            filename=filename,
            size=size,
            sha256=_require_non_empty_string(
                payload.get("sha256"),
                f"asset {filename!r} sha256",
            ),
            content_encoding=_optional_string(
                payload.get("content_encoding"),
                f"asset {filename!r} content_encoding",
            ),
            content_type=_require_non_empty_string(
                payload.get("content_type"),
                f"asset {filename!r} content_type",
            ),
        )


@dataclass(frozen=True)
class DeltaInfo:
    base_version: str | None
    requires_exact_base: bool
    operations: int
    metadata_operations: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_version": self.base_version,
            "requires_exact_base": self.requires_exact_base,
            "operations": self.operations,
            "metadata_operations": self.metadata_operations,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DeltaInfo:
        base_version = payload.get("base_version")
        if base_version is not None and not isinstance(base_version, str):
            raise ValidationError("delta.base_version must be a string or null")
        requires_exact_base = payload.get("requires_exact_base")
        if not isinstance(requires_exact_base, bool):
            raise ValidationError("delta.requires_exact_base must be a boolean")
        operations = payload.get("operations")
        if not isinstance(operations, int) or operations < 0:
            raise ValidationError("delta.operations must be a non-negative integer")
        metadata_operations = payload.get("metadata_operations")
        if not isinstance(metadata_operations, int) or metadata_operations < 0:
            raise ValidationError("delta.metadata_operations must be a non-negative integer")
        if base_version is None and (
            requires_exact_base or operations != 0 or metadata_operations != 0
        ):
            raise ValidationError("seed delta must be empty and not require a base")
        if base_version is not None and not requires_exact_base:
            raise ValidationError("versioned delta must require its exact base")
        return cls(
            base_version=base_version,
            requires_exact_base=requires_exact_base,
            operations=operations,
            metadata_operations=metadata_operations,
        )


@dataclass(frozen=True)
class CatalogManifest:
    schema_version: int
    catalog_key: str
    version: str
    previous_version: str | None
    embedding_model: str
    source_revision: SourceRevision
    descriptor: CatalogDescriptor
    rows: int
    dim: int
    dtype: str
    assets: dict[str, AssetInfo]
    delta: DeltaInfo

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "catalog_key": self.catalog_key,
            "version": self.version,
            "previous_version": self.previous_version,
            "embedding_model": self.embedding_model,
            "source_revision": self.source_revision.to_dict(),
            "descriptor": self.descriptor.to_dict(),
            "rows": self.rows,
            "dim": self.dim,
            "dtype": self.dtype,
            "assets": {name: asset.to_dict() for name, asset in sorted(self.assets.items())},
            "delta": self.delta.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CatalogManifest:
        schema_version = payload.get("schema_version")
        if schema_version != 2:
            raise ValidationError("manifest schema_version must be 2")
        rows = payload.get("rows")
        if not isinstance(rows, int) or rows <= 0:
            raise ValidationError("rows must be a positive integer")
        dim = payload.get("dim")
        if not isinstance(dim, int) or dim <= 0:
            raise ValidationError("dim must be a positive integer")
        dtype = payload.get("dtype")
        if dtype != "float16":
            raise ValidationError("dtype must be 'float16'")
        previous_version = payload.get("previous_version")
        if previous_version is not None and not isinstance(previous_version, str):
            raise ValidationError("previous_version must be a string or null")
        delta = DeltaInfo.from_dict(_require_mapping(payload.get("delta"), "delta"))
        if previous_version != delta.base_version:
            raise ValidationError("previous_version must match delta.base_version")
        assets = {
            _require_non_empty_string(name, "asset key"): AssetInfo.from_dict(
                _require_mapping(asset_payload, f"assets[{name!r}]")
            )
            for name, asset_payload in _require_mapping(payload.get("assets"), "assets").items()
        }
        missing_assets = _REQUIRED_ASSET_NAMES.difference(assets)
        if missing_assets:
            raise ValidationError(f"manifest is missing required assets: {sorted(missing_assets)}")
        if delta.operations and "identifiers_delta" not in assets:
            raise ValidationError("manifest with identifier delta operations needs an asset")
        if not delta.operations and {
            "identifiers_delta",
            "embeddings_delta",
        }.intersection(assets):
            raise ValidationError("manifest without identifier delta operations cannot have assets")
        if "embeddings_delta" in assets and "identifiers_delta" not in assets:
            raise ValidationError("embedding delta requires identifier delta operations")
        if delta.metadata_operations and "metadata_delta" not in assets:
            raise ValidationError("manifest with metadata delta operations needs an asset")
        if not delta.metadata_operations and "metadata_delta" in assets:
            raise ValidationError("manifest without metadata delta operations cannot have an asset")
        return cls(
            schema_version=schema_version,
            catalog_key=_require_non_empty_string(payload.get("catalog_key"), "catalog_key"),
            version=_require_non_empty_string(payload.get("version"), "version"),
            previous_version=previous_version,
            embedding_model=_require_non_empty_string(
                payload.get("embedding_model"),
                "embedding_model",
            ),
            source_revision=SourceRevision.from_dict(
                _require_mapping(payload.get("source_revision"), "source_revision")
            ),
            descriptor=CatalogDescriptor.from_dict(
                _require_mapping(payload.get("descriptor"), "descriptor")
            ),
            rows=rows,
            dim=dim,
            dtype=dtype,
            assets=assets,
            delta=delta,
        )


@dataclass(frozen=True)
class DeltaUpsert:
    row: RecognitionRow
    embedding_index: int


@dataclass(frozen=True)
class DeltaDelete:
    key: str


DeltaOperation = DeltaUpsert | DeltaDelete


@dataclass(frozen=True)
class MetadataUpsert:
    key: str
    metadata: dict[str, JSONValue]


@dataclass(frozen=True)
class MetadataDelete:
    key: str


MetadataDeltaOperation = MetadataUpsert | MetadataDelete


@dataclass(frozen=True)
class DeltaBundle:
    operations: tuple[DeltaOperation, ...]
    embeddings: NDArray[np.float16]
    metadata_operations: tuple[MetadataDeltaOperation, ...]


@dataclass(frozen=True)
class CatalogBuild:
    manifest: CatalogManifest
    rows: tuple[RecognitionRow, ...]
    embeddings: NDArray[np.float16]
    state: dict[str, StateRecord]


def build_catalog(
    rows: Iterable[RecognitionRow],
    embedder: Embedder,
    output_dir: str | Path,
    catalog_key: str,
    version: str,
    embedding_model: str,
    source_revision: SourceRevision,
    descriptor: CatalogDescriptor,
    seed_embeddings: Mapping[str, ArrayLike] | None = None,
    previous_build: CatalogBuild | None = None,
    image_loader: ImageLoader | None = None,
    batch_size: int = 16,
) -> CatalogBuild:
    if batch_size <= 0:
        raise ValidationError("batch_size must be a positive integer")
    catalog_key = _require_non_empty_string(catalog_key, "catalog_key")
    version = _require_non_empty_string(version, "version")
    embedding_model = _require_non_empty_string(embedding_model, "embedding_model")
    if not isinstance(descriptor, CatalogDescriptor):
        raise ValidationError("descriptor must be a CatalogDescriptor")
    if not isinstance(source_revision, SourceRevision):
        raise ValidationError("source_revision must be a SourceRevision")
    normalized_rows = _prepare_rows(rows, descriptor)
    if not normalized_rows:
        raise ValidationError("source rows must not be empty")
    if previous_build is not None and seed_embeddings is not None:
        raise ValidationError("seed_embeddings cannot be combined with previous_build")
    if previous_build is not None:
        _validate_previous_build(
            previous_build,
            catalog_key=catalog_key,
            embedding_model=embedding_model,
            descriptor=descriptor,
        )
    image_loader = image_loader or default_image_loader
    embeddings = _build_embeddings(
        normalized_rows,
        embedder=embedder,
        image_loader=image_loader,
        previous_build=previous_build,
        seed_embeddings=seed_embeddings,
        batch_size=batch_size,
    )
    delta_operations, delta_embeddings, metadata_operations = _build_deltas(
        normalized_rows,
        embeddings,
        previous_build,
    )
    catalog_slug = catalog_key_to_slug(catalog_key)
    asset_payloads = _build_asset_payloads(
        catalog_slug=catalog_slug,
        rows=normalized_rows,
        embeddings=embeddings,
        delta_operations=delta_operations,
        delta_embeddings=delta_embeddings,
        metadata_operations=metadata_operations,
    )
    manifest = CatalogManifest(
        schema_version=2,
        catalog_key=catalog_key,
        version=version,
        previous_version=None if previous_build is None else previous_build.manifest.version,
        embedding_model=embedding_model,
        source_revision=source_revision,
        descriptor=descriptor,
        rows=len(normalized_rows),
        dim=int(embeddings.shape[1]),
        dtype="float16",
        assets={
            name: _asset_info(filename=filename, payload=payload, content_type=content_type)
            for name, (filename, content_type, payload) in asset_payloads.items()
        },
        delta=DeltaInfo(
            base_version=None if previous_build is None else previous_build.manifest.version,
            requires_exact_base=previous_build is not None,
            operations=len(delta_operations),
            metadata_operations=len(metadata_operations),
        ),
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for filename, _, payload in asset_payloads.values():
        (output_path / filename).write_bytes(payload)
    manifest_filename = manifest_filename_for_catalog(catalog_key)
    (output_path / manifest_filename).write_bytes(_canonical_json_bytes(manifest.to_dict()))
    state_by_key = {row.key: row.state_record() for row in normalized_rows}
    return CatalogBuild(
        manifest=manifest,
        rows=tuple(normalized_rows),
        embeddings=embeddings,
        state=state_by_key,
    )


def manifest_filename_for_catalog(catalog_key: str) -> str:
    return f"{catalog_key_to_slug(catalog_key)}.manifest.json"


def catalog_key_to_slug(catalog_key: str) -> str:
    normalized_key = _require_non_empty_string(catalog_key, "catalog_key")
    parts = normalized_key.replace("\\", "/").split("/")
    slug_parts: list[str] = []
    for index, part in enumerate(parts):
        clean_part = _SAFE_SLUG_CHARS.sub("-", part).strip("-.")
        if not clean_part:
            raise ValidationError(
                f"catalog_key contains an empty or unsafe path segment at position {index}"
            )
        slug_parts.append(clean_part)
    slug = "--".join(slug_parts)
    if not slug or slug in {".", ".."}:
        raise ValidationError("catalog_key cannot be converted to a safe filename slug")
    return slug


def load_manifest(manifest_path: str | Path) -> CatalogManifest:
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    return CatalogManifest.from_dict(_require_mapping(payload, "manifest"))


def load_catalog_build(
    manifest_path: str | Path,
    asset_dir: str | Path | None = None,
) -> CatalogBuild:
    manifest = load_manifest(manifest_path)
    asset_root = Path(asset_dir) if asset_dir is not None else Path(manifest_path).resolve().parent
    minimal_rows = _load_jsonl_asset(manifest, asset_root, "identifiers")
    metadata_rows = _load_jsonl_asset(manifest, asset_root, "metadata")
    state_rows = _load_jsonl_asset(manifest, asset_root, "state_rows")
    embeddings = _load_embedding_asset(
        manifest,
        asset_root,
        asset_name="embeddings",
        expected_rows=manifest.rows,
    )
    if len(minimal_rows) != manifest.rows:
        raise ValidationError(
            f"manifest rows={manifest.rows} does not match recognition row count "
            f"{len(minimal_rows)}"
        )
    if len(state_rows) != manifest.rows:
        raise ValidationError("state rows must be line-aligned with identifiers")
    if len(metadata_rows) != manifest.rows:
        raise ValidationError("metadata rows must be line-aligned with identifiers")
    rows: list[RecognitionRow] = []
    seen_keys: set[str] = set()
    state_by_key: dict[str, StateRecord] = {}
    for index, (payload, state_payload, metadata_payload) in enumerate(
        zip(minimal_rows, state_rows, metadata_rows, strict=True)
    ):
        minimal_payload = _require_mapping(payload, f"identifiers line {index + 1}")
        state_mapping = _require_mapping(state_payload, f"state_rows line {index + 1}")
        if metadata_payload is not None and not isinstance(metadata_payload, Mapping):
            raise ValidationError(f"metadata line {index + 1} must be an object or null")
        row = RecognitionRow.from_artifact_records(
            minimal_payload,
            state_mapping,
            provider=manifest.descriptor.source,
            metadata=metadata_payload,
        )
        key = row.key
        if key in seen_keys:
            raise ValidationError(f"duplicate recognition entry for key {key!r}")
        seen_keys.add(key)
        _validate_result_identifier(row, manifest.descriptor, "recognition row")
        rows.append(row)
        state_by_key[key] = row.state_record()
    if embeddings.shape != (manifest.rows, manifest.dim):
        raise ValidationError(
            "embedding matrix shape "
            f"{embeddings.shape} does not match ({manifest.rows}, {manifest.dim})"
        )
    return CatalogBuild(
        manifest=manifest,
        rows=tuple(rows),
        embeddings=embeddings,
        state=state_by_key,
    )


def load_delta_bundle(
    manifest_path: str | Path,
    asset_dir: str | Path | None = None,
) -> DeltaBundle:
    manifest = load_manifest(manifest_path)
    asset_root = Path(asset_dir) if asset_dir is not None else Path(manifest_path).resolve().parent
    operation_payloads = (
        _load_jsonl_asset(manifest, asset_root, "identifiers_delta")
        if manifest.delta.operations
        else []
    )
    operations: list[DeltaOperation] = []
    seen_keys: set[str] = set()
    upsert_count = 0
    for raw_payload in operation_payloads:
        payload = _require_mapping(raw_payload, "identifier delta operation")
        op_type = _require_non_empty_string(payload.get("op"), "delta op")
        if op_type == "delete":
            primary_id, face_index = _parse_identity_target(payload, "delta delete")
            _require_allowed_fields(
                payload,
                required={"op", "id"},
                allowed={"op", "id", "face_index"},
                name="delta delete",
            )
            key = catalog_row_key(manifest.descriptor.source, primary_id, face_index)
            if key in seen_keys:
                raise ValidationError(f"duplicate delta operation for key {key!r}")
            seen_keys.add(key)
            operations.append(DeltaDelete(key=key))
            continue
        if op_type != "upsert":
            raise ValidationError(f"unsupported delta op {op_type!r}")
        _require_exact_fields(
            payload,
            {"op", "record", "state", "embedding_index"},
            "delta upsert",
        )
        record = _require_mapping(payload.get("record"), "delta upsert record")
        state_payload = _require_mapping(payload.get("state"), "delta upsert state")
        embedding_index = payload.get("embedding_index")
        if not isinstance(embedding_index, int) or embedding_index < 0:
            raise ValidationError("delta upsert embedding_index must be a non-negative integer")
        row = RecognitionRow.from_artifact_records(
            record,
            state_payload,
            provider=manifest.descriptor.source,
        )
        _validate_result_identifier(row, manifest.descriptor, "delta upsert")
        if row.key in seen_keys:
            raise ValidationError(f"duplicate delta operation for key {row.key!r}")
        seen_keys.add(row.key)
        operations.append(DeltaUpsert(row=row, embedding_index=embedding_index))
        upsert_count += 1
    embeddings = (
        _load_embedding_asset(
            manifest,
            asset_root,
            asset_name="embeddings_delta",
            expected_rows=upsert_count,
        )
        if upsert_count
        else np.empty((0, manifest.dim), dtype=np.float16)
    )
    if embeddings.shape != (upsert_count, manifest.dim):
        raise ValidationError(
            "delta embedding matrix shape "
            f"{embeddings.shape} does not match ({upsert_count}, {manifest.dim})"
        )
    metadata_operations = (
        _load_metadata_delta(
            _load_jsonl_asset(manifest, asset_root, "metadata_delta"),
            provider=manifest.descriptor.source,
        )
        if manifest.delta.metadata_operations
        else []
    )
    if len(operations) != manifest.delta.operations:
        raise ValidationError(
            "manifest delta.operations="
            f"{manifest.delta.operations} does not match {len(operations)}"
        )
    if len(metadata_operations) != manifest.delta.metadata_operations:
        raise ValidationError(
            "manifest delta.metadata_operations="
            f"{manifest.delta.metadata_operations} does not match {len(metadata_operations)}"
        )
    return DeltaBundle(
        operations=tuple(operations),
        embeddings=embeddings,
        metadata_operations=tuple(metadata_operations),
    )


def apply_delta(
    previous_build: CatalogBuild | None,
    manifest_path: str | Path,
    asset_dir: str | Path | None = None,
) -> CatalogBuild:
    manifest = load_manifest(manifest_path)
    if manifest.delta.base_version is None:
        raise ValidationError("seed releases must be installed from the full snapshot")
    else:
        if previous_build is None:
            raise ValidationError("previous_build is required for a versioned delta")
        _validate_previous_build(
            previous_build,
            catalog_key=manifest.catalog_key,
            embedding_model=manifest.embedding_model,
            descriptor=manifest.descriptor,
        )
        if previous_build.manifest.version != manifest.delta.base_version:
            raise ValidationError(
                "previous build version does not match delta base_version "
                f"({previous_build.manifest.version!r} != {manifest.delta.base_version!r})"
            )
        if previous_build.embeddings.shape[1] != manifest.dim:
            raise ValidationError(
                "previous build embedding dimension does not match delta dimension"
            )
    delta = load_delta_bundle(manifest_path, asset_dir=asset_dir)
    current_rows: dict[str, RecognitionRow] = {}
    current_embeddings: dict[str, NDArray[np.float16]] = {}
    current_metadata: dict[str, dict[str, JSONValue]] = {}
    if previous_build is not None:
        for index, row in enumerate(previous_build.rows):
            current_rows[row.key] = row.with_metadata(None)
            current_embeddings[row.key] = previous_build.embeddings[index].copy()
            if row.metadata is not None:
                current_metadata[row.key] = row.metadata
    for operation in delta.operations:
        if isinstance(operation, DeltaDelete):
            if operation.key not in current_rows:
                raise ValidationError(f"delta delete references missing key {operation.key!r}")
            current_rows.pop(operation.key)
            current_embeddings.pop(operation.key)
            current_metadata.pop(operation.key, None)
            continue
        if operation.embedding_index >= len(delta.embeddings):
            raise ValidationError(
                f"delta upsert for key {operation.row.key!r} references missing embedding index "
                f"{operation.embedding_index}"
            )
        current_rows[operation.row.key] = operation.row.with_metadata(None)
        current_embeddings[operation.row.key] = delta.embeddings[operation.embedding_index].copy()
    for operation in delta.metadata_operations:
        if isinstance(operation, MetadataDelete):
            current_metadata.pop(operation.key, None)
            continue
        if operation.key not in current_rows:
            raise ValidationError(
                f"metadata delta references missing recognition key {operation.key!r}"
            )
        current_metadata[operation.key] = operation.metadata
    sorted_keys = sorted(current_rows)
    rows: list[RecognitionRow] = []
    embedding_rows: list[NDArray[np.float16]] = []
    state_by_key: dict[str, StateRecord] = {}
    for key in sorted_keys:
        row = current_rows[key].with_metadata(current_metadata.get(key))
        rows.append(row)
        embedding_rows.append(current_embeddings[key])
        state_by_key[key] = row.state_record()
    if len(rows) != manifest.rows:
        raise ValidationError(
            f"delta reconstructed {len(rows)} rows but manifest expects {manifest.rows}"
        )
    embeddings = np.vstack(embedding_rows).astype(np.float16, copy=False)
    if embeddings.shape != (manifest.rows, manifest.dim):
        raise ValidationError(
            f"delta reconstructed embeddings shape {embeddings.shape} does not match "
            f"({manifest.rows}, {manifest.dim})"
        )
    return CatalogBuild(
        manifest=manifest,
        rows=tuple(rows),
        embeddings=embeddings,
        state=state_by_key,
    )


def default_image_loader(image_url: str) -> Image.Image:
    with urlopen(image_url, timeout=30) as response:
        payload = response.read()
    with Image.open(io.BytesIO(payload)) as image:
        loaded = image.convert("RGB")
        loaded.load()
    return loaded


def validate_artifacts(
    manifest_path: str | Path,
    asset_dir: str | Path | None = None,
    previous_build: CatalogBuild | None = None,
) -> CatalogBuild:
    build = load_catalog_build(manifest_path, asset_dir=asset_dir)
    load_delta_bundle(manifest_path, asset_dir=asset_dir)
    if previous_build is not None:
        reconstructed = apply_delta(previous_build, manifest_path, asset_dir=asset_dir)
        if not _catalog_builds_equal(build, reconstructed):
            raise ValidationError("delta application did not reconstruct the current catalog build")
    return build


def _catalog_builds_equal(left: CatalogBuild, right: CatalogBuild) -> bool:
    return (
        [row for row in left.rows] == [row for row in right.rows]
        and left.state == right.state
        and np.array_equal(left.embeddings, right.embeddings)
        and left.manifest.to_dict() == right.manifest.to_dict()
    )


def _validate_previous_build(
    previous_build: CatalogBuild,
    *,
    catalog_key: str,
    embedding_model: str,
    descriptor: CatalogDescriptor | None = None,
) -> None:
    if previous_build.manifest.catalog_key != catalog_key:
        raise ValidationError("previous build catalog_key does not match current catalog_key")
    if previous_build.manifest.dtype != "float16":
        raise ValidationError("previous build dtype must be float16")
    if previous_build.manifest.embedding_model != embedding_model:
        raise ValidationError(
            "previous build embedding_model does not match current embedding_model"
        )
    if descriptor is not None and previous_build.manifest.descriptor != descriptor:
        raise ValidationError("previous build descriptor does not match current descriptor")


def _validate_result_identifier(
    row: RecognitionRow,
    descriptor: CatalogDescriptor,
    record_type: str,
) -> None:
    if row.provider != descriptor.source:
        raise ValidationError(
            f"{record_type} {row.key!r} provider {row.provider!r} does not match "
            f"descriptor source {descriptor.source!r}"
        )
    expected_result_identifier = primary_identifier_for_provider(row.provider)
    if (
        expected_result_identifier is not None
        and descriptor.result_identifier != expected_result_identifier
    ):
        raise ValidationError(
            f"{record_type} {row.key!r} provider {row.provider!r} requires result identifier "
            f"{expected_result_identifier!r}"
        )
    if descriptor.result_identifier in row.identifiers:
        raise ValidationError(
            f"{record_type} {row.key!r} duplicates primary result identifier "
            f"{descriptor.result_identifier!r} in peer identifiers"
        )


def _prepare_rows(
    rows: Iterable[RecognitionRow],
    descriptor: CatalogDescriptor,
) -> list[RecognitionRow]:
    normalized: list[RecognitionRow] = []
    seen_keys: set[str] = set()
    for raw_row in rows:
        row = RecognitionRow.from_artifact_records(
            raw_row.minimal_record(),
            raw_row.state_record(),
            provider=raw_row.provider,
            metadata=raw_row.metadata,
        )
        _validate_result_identifier(row, descriptor, "source row")
        if row.key in seen_keys:
            raise ValidationError(f"duplicate key {row.key!r}")
        seen_keys.add(row.key)
        normalized.append(row)
    normalized.sort(key=lambda row: row.key)
    return normalized


def _build_embeddings(
    rows: Sequence[RecognitionRow],
    embedder: Embedder,
    image_loader: ImageLoader,
    previous_build: CatalogBuild | None,
    seed_embeddings: Mapping[str, ArrayLike] | None,
    batch_size: int,
) -> NDArray[np.float16]:
    expected_dim = None if previous_build is None else previous_build.embeddings.shape[1]
    previous_lookup: dict[str, int] = {}
    if previous_build is not None:
        previous_lookup = {row.key: index for index, row in enumerate(previous_build.rows)}
    seed_lookup, expected_dim = _validate_seed_embeddings(
        rows,
        seed_embeddings=seed_embeddings,
        expected_dim=expected_dim,
    )
    slots: list[NDArray[np.float16] | None] = [None] * len(rows)
    pending_indexes: list[int] = []
    for index, row in enumerate(rows):
        seed_embedding = seed_lookup.get(row.key)
        if seed_embedding is not None:
            slots[index] = seed_embedding.astype(np.float16, copy=True)
            continue
        previous_index = previous_lookup.get(row.key)
        if previous_index is None:
            pending_indexes.append(index)
            continue
        previous_row = previous_build.rows[previous_index]
        if previous_row.image_fingerprint == row.image_fingerprint:
            slots[index] = previous_build.embeddings[previous_index].astype(np.float16, copy=True)
        else:
            pending_indexes.append(index)
    for start in range(0, len(pending_indexes), batch_size):
        batch_indexes = pending_indexes[start : start + batch_size]
        images = [image_loader(rows[index].image_url) for index in batch_indexes]
        try:
            payload = embedder(images)
        finally:
            for image in images:
                image.close()
        batch_embeddings = _validate_embedding_batch(
            payload,
            expected_rows=len(batch_indexes),
            expected_dim=expected_dim,
        )
        expected_dim = int(batch_embeddings.shape[1])
        for row_index, embedding in zip(batch_indexes, batch_embeddings, strict=True):
            slots[row_index] = embedding.astype(np.float16, copy=False)
    if expected_dim is None:
        raise ValidationError("unable to determine embedding dimension")
    missing_slots = [rows[index].key for index, item in enumerate(slots) if item is None]
    if missing_slots:
        raise ValidationError(f"missing embeddings for keys: {missing_slots}")
    embeddings = np.vstack([item for item in slots if item is not None]).astype(
        np.float16,
        copy=False,
    )
    if embeddings.shape != (len(rows), expected_dim):
        raise ValidationError(
            "embedding matrix shape "
            f"{embeddings.shape} does not match ({len(rows)}, {expected_dim})"
        )
    return embeddings


def _validate_embedding_batch(
    payload: Sequence[Sequence[float]] | NDArray[np.floating[Any]],
    expected_rows: int,
    expected_dim: int | None,
) -> NDArray[np.float32]:
    embeddings = np.asarray(payload, dtype=np.float32)
    return _validate_embedding_array(
        embeddings,
        expected_rows=expected_rows,
        expected_dim=expected_dim,
        source_name="embedder",
    )


def _validate_seed_embeddings(
    rows: Sequence[RecognitionRow],
    seed_embeddings: Mapping[str, ArrayLike] | None,
    expected_dim: int | None,
) -> tuple[dict[str, NDArray[np.float32]], int | None]:
    if seed_embeddings is None:
        return {}, expected_dim
    row_keys = {row.key for row in rows}
    unknown_keys = sorted(set(seed_embeddings).difference(row_keys))
    if unknown_keys:
        raise ValidationError(f"seed_embeddings contain unknown keys: {unknown_keys}")
    validated: dict[str, NDArray[np.float32]] = {}
    current_dim = expected_dim
    for key, raw_embedding in seed_embeddings.items():
        embedding = np.asarray(raw_embedding, dtype=np.float32)
        validated_embedding = _validate_embedding_array(
            embedding,
            expected_rows=1,
            expected_dim=current_dim,
            source_name=f"seed_embeddings[{key!r}]",
        )[0]
        if current_dim is None:
            current_dim = int(validated_embedding.shape[0])
        validated[key] = validated_embedding
    return validated, current_dim


def _validate_embedding_array(
    embeddings: NDArray[np.float32],
    expected_rows: int,
    expected_dim: int | None,
    source_name: str,
) -> NDArray[np.float32]:
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    if embeddings.ndim != 2:
        raise ValidationError(_embedding_error(source_name, "must be a 1D or 2D array-like batch"))
    if embeddings.shape[0] != expected_rows:
        raise ValidationError(
            _embedding_error(
                source_name,
                f"returned {embeddings.shape[0]} rows but expected {expected_rows}",
            )
        )
    if embeddings.shape[1] <= 0:
        raise ValidationError(_embedding_error(source_name, "returned zero-dimensional embeddings"))
    if expected_dim is not None and embeddings.shape[1] != expected_dim:
        raise ValidationError(
            _embedding_error(
                source_name,
                f"returned dimension {embeddings.shape[1]} but expected {expected_dim}",
            )
        )
    if not np.isfinite(embeddings).all():
        raise ValidationError(
            _embedding_error(source_name, "embeddings must contain only finite values")
        )
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=5e-3):
        raise ValidationError(_embedding_error(source_name, "embeddings must be L2-normalized"))
    return embeddings


def _embedding_error(source_name: str, message: str) -> str:
    return message if source_name == "embedder" else f"{source_name} {message}"


def _build_deltas(
    rows: Sequence[RecognitionRow],
    embeddings: NDArray[np.float16],
    previous_build: CatalogBuild | None,
) -> tuple[list[dict[str, Any]], NDArray[np.float16], list[dict[str, Any]]]:
    if previous_build is None:
        return [], np.empty((0, embeddings.shape[1]), dtype=np.float16), []
    previous_rows = {row.key: row for row in previous_build.rows}
    delta_operations: list[dict[str, Any]] = []
    delta_embedding_rows: list[NDArray[np.float16]] = []
    previous_keys = set(previous_rows)
    current_rows = {row.key: row for row in rows}
    current_keys = set(current_rows)
    for key in sorted(previous_keys - current_keys):
        previous_row = previous_rows[key]
        delta_operations.append({"op": "delete", **_identity_target(previous_row)})
    for index, row in enumerate(rows):
        previous_row = previous_rows.get(row.key)
        if previous_row is None or _requires_delta_upsert(previous_row, row):
            delta_operations.append(
                {
                    "op": "upsert",
                    "record": row.minimal_record(),
                    "state": row.state_record().to_dict(),
                    "embedding_index": len(delta_embedding_rows),
                }
            )
            delta_embedding_rows.append(embeddings[index])
    previous_metadata = {
        row.key: row.metadata for row in previous_rows.values() if row.metadata is not None
    }
    current_metadata = {row.key: row.metadata for row in rows if row.metadata is not None}
    metadata_operations: list[dict[str, Any]] = []
    for key in sorted(set(previous_metadata) - set(current_metadata)):
        metadata_operations.append(
            {"op": "delete", **_identity_target(previous_rows[key])}
        )
    for key in sorted(current_metadata):
        if previous_metadata.get(key) != current_metadata[key]:
            metadata_operations.append(
                {
                    "op": "upsert",
                    **_identity_target(current_rows[key]),
                    "metadata": current_metadata[key],
                }
            )
    if delta_embedding_rows:
        delta_embeddings = np.vstack(delta_embedding_rows).astype(np.float16, copy=False)
    else:
        delta_embeddings = np.empty((0, embeddings.shape[1]), dtype=np.float16)
    return delta_operations, delta_embeddings, metadata_operations


def _requires_delta_upsert(previous_row: RecognitionRow, current_row: RecognitionRow) -> bool:
    return (
        previous_row.minimal_record() != current_row.minimal_record()
        or previous_row.state_record() != current_row.state_record()
    )


def _build_asset_payloads(
    catalog_slug: str,
    rows: Sequence[RecognitionRow],
    embeddings: NDArray[np.float16],
    delta_operations: Sequence[dict[str, Any]],
    delta_embeddings: NDArray[np.float16],
    metadata_operations: Sequence[dict[str, Any]],
) -> dict[str, tuple[str, str, bytes]]:
    assets = {
        "embeddings": (
            f"{catalog_slug}.embeddings.f16.gz",
            "application/octet-stream",
            _gzip_bytes(_embedding_bytes(embeddings)),
        ),
        "identifiers": (
            f"{catalog_slug}.identifiers.jsonl.gz",
            "application/x-ndjson",
            _gzip_bytes(_jsonl_bytes(row.minimal_record() for row in rows)),
        ),
        "metadata": (
            f"{catalog_slug}.metadata.jsonl.gz",
            "application/x-ndjson",
            _gzip_bytes(_jsonl_bytes(_iter_metadata_records(rows))),
        ),
        "state_rows": (
            f"{catalog_slug}.state.jsonl.gz",
            "application/x-ndjson",
            _gzip_bytes(_jsonl_bytes(row.state_record().to_dict() for row in rows)),
        ),
    }
    if delta_operations:
        assets["identifiers_delta"] = (
            f"{catalog_slug}.identifiers.delta.jsonl.gz",
            "application/x-ndjson",
            _gzip_bytes(_jsonl_bytes(delta_operations)),
        )
    if len(delta_embeddings):
        assets["embeddings_delta"] = (
            f"{catalog_slug}.embeddings.delta.f16.gz",
            "application/octet-stream",
            _gzip_bytes(_embedding_bytes(delta_embeddings)),
        )
    if metadata_operations:
        assets["metadata_delta"] = (
            f"{catalog_slug}.metadata.delta.jsonl.gz",
            "application/x-ndjson",
            _gzip_bytes(_jsonl_bytes(metadata_operations)),
        )
    return assets


def _embedding_bytes(embeddings: NDArray[np.float16]) -> bytes:
    return np.asarray(embeddings, dtype="<f2").tobytes(order="C")


def _asset_info(filename: str, payload: bytes, content_type: str) -> AssetInfo:
    return AssetInfo(
        filename=filename,
        size=len(payload),
        sha256=sha256(payload).hexdigest(),
        content_encoding="gzip",
        content_type=content_type,
    )


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _jsonl_bytes(records: Iterable[Any]) -> bytes:
    return b"".join(_canonical_json_bytes(record) + b"\n" for record in records)


def _gzip_bytes(payload: bytes) -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(
        filename="",
        mode="wb",
        fileobj=output,
        compresslevel=9,
        mtime=0,
    ) as compressed:
        compressed.write(payload)
    return output.getvalue()


def _load_jsonl_asset(
    manifest: CatalogManifest,
    asset_root: Path,
    asset_name: str,
) -> list[Any]:
    decoded = _read_decoded_gzip_asset(manifest, asset_root, asset_name)
    if not decoded:
        return []
    records: list[Any] = []
    for line_number, line in enumerate(decoded.decode("utf-8").splitlines(), start=1):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise AssetIntegrityError(
                f"{asset_name} line {line_number} is not valid JSON"
            ) from error
    return records


def _load_embedding_asset(
    manifest: CatalogManifest,
    asset_root: Path,
    asset_name: str,
    expected_rows: int,
) -> NDArray[np.float16]:
    decoded = _read_decoded_gzip_asset(manifest, asset_root, asset_name)
    embeddings = np.frombuffer(decoded, dtype="<f2")
    expected_values = expected_rows * manifest.dim
    if embeddings.size != expected_values:
        raise AssetIntegrityError(
            f"{asset_name} contains {embeddings.size} values but expected {expected_values}"
        )
    return embeddings.reshape((expected_rows, manifest.dim)).astype(np.float16, copy=False)


def _read_decoded_gzip_asset(
    manifest: CatalogManifest,
    asset_root: Path,
    asset_name: str,
) -> bytes:
    asset = manifest.assets[asset_name]
    if asset.content_encoding != "gzip":
        raise ValidationError(f"asset {asset.filename!r} must use gzip content encoding")
    raw_bytes = _read_verified_asset(asset_root / asset.filename, asset)
    try:
        return gzip.decompress(raw_bytes)
    except (OSError, EOFError) as error:
        raise AssetIntegrityError(f"unable to decode gzip asset {asset.filename!r}") from error


def _read_verified_asset(path: Path, asset: AssetInfo) -> bytes:
    if not path.exists():
        raise AssetIntegrityError(f"missing asset file {path}")
    size = path.stat().st_size
    if size != asset.size:
        raise AssetIntegrityError(
            f"asset {asset.filename!r} size mismatch: expected {asset.size}, found {size}"
        )
    payload = path.read_bytes()
    digest = sha256(payload).hexdigest()
    if digest != asset.sha256:
        raise AssetIntegrityError(
            f"asset {asset.filename!r} sha256 mismatch: expected {asset.sha256}, found {digest}"
        )
    return payload


def _load_metadata_delta(
    rows: Sequence[Any],
    *,
    provider: str,
) -> list[MetadataDeltaOperation]:
    metadata_operations: list[MetadataDeltaOperation] = []
    seen_keys: set[str] = set()
    for raw_payload in rows:
        payload = _require_mapping(raw_payload, "metadata delta operation")
        op_type = _require_non_empty_string(payload.get("op"), "metadata delta op")
        primary_id, face_index = _parse_identity_target(payload, "metadata delta")
        key = catalog_row_key(provider, primary_id, face_index)
        if key in seen_keys:
            raise ValidationError(f"duplicate metadata delta operation for key {key!r}")
        seen_keys.add(key)
        if op_type == "delete":
            _require_allowed_fields(
                payload,
                required={"op", "id"},
                allowed={"op", "id", "face_index"},
                name="metadata delta delete",
            )
            metadata_operations.append(MetadataDelete(key=key))
        elif op_type == "upsert":
            _require_allowed_fields(
                payload,
                required={"op", "id", "metadata"},
                allowed={"op", "id", "face_index", "metadata"},
                name="metadata delta upsert",
            )
            metadata_operations.append(
                MetadataUpsert(
                    key=key,
                    metadata=_canonicalize_metadata(
                        _require_mapping(payload.get("metadata"), f"metadata delta {key!r}")
                    ),
                )
            )
        else:
            raise ValidationError(f"unsupported metadata delta op {op_type!r}")
    return metadata_operations


def _iter_metadata_records(
    rows: Sequence[RecognitionRow],
) -> Iterable[dict[str, JSONValue] | None]:
    for row in rows:
        yield row.metadata


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{name} must be a mapping")
    return value


def _require_allowed_fields(
    payload: Mapping[str, Any],
    *,
    required: set[str],
    allowed: set[str],
    name: str,
) -> None:
    missing = required.difference(payload)
    extra = set(payload).difference(allowed)
    if missing or extra:
        raise ValidationError(
            f"{name} fields must include {sorted(required)} and be limited to "
            f"{sorted(allowed)}"
        )


def _require_exact_fields(
    payload: Mapping[str, Any],
    expected: set[str],
    name: str,
) -> None:
    if set(payload) != expected:
        raise ValidationError(f"{name} fields must be exactly {sorted(expected)}")


def catalog_row_key(provider: str, primary_id: str, face_index: int = 0) -> str:
    provider = _identity_component(provider, "provider")
    primary_id = _identity_component(primary_id, "id")
    if not isinstance(face_index, int) or isinstance(face_index, bool) or face_index < 0:
        raise ValidationError("face_index must be a non-negative integer")
    base = f"{provider}:{primary_id}"
    return base if face_index == 0 else f"{base}:face:{face_index}"


def primary_identifier_for_provider(provider: str) -> str | None:
    return _PRIMARY_IDENTIFIERS.get(provider)


def _identity_target(row: RecognitionRow) -> dict[str, Any]:
    target: dict[str, Any] = {"id": row.id}
    if row.face_index:
        target["face_index"] = row.face_index
    return target


def _parse_identity_target(
    payload: Mapping[str, Any],
    name: str,
) -> tuple[str, int]:
    primary_id = _identity_component(payload.get("id"), f"{name} id")
    face_index = payload.get("face_index", 0)
    if (
        not isinstance(face_index, int)
        or isinstance(face_index, bool)
        or face_index < 0
    ):
        raise ValidationError(f"{name} face_index must be a non-negative integer")
    if face_index == 0 and "face_index" in payload:
        raise ValidationError(f"{name} must omit face_index for face 0")
    return primary_id, face_index


def _identity_component(value: Any, name: str) -> str:
    component = _require_non_empty_string(value, name)
    if ":" in component:
        raise ValidationError(f"{name} must not contain ':'")
    return component


def _require_non_empty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{name} must be a non-empty string")
    return value.strip()


def _optional_string(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValidationError(f"{name} must be a string if set")
    return value


def _canonicalize_metadata(metadata: Mapping[str, Any]) -> dict[str, JSONValue]:
    return {
        _require_non_empty_string(key, "metadata key"): _canonicalize_json_value(value)
        for key, value in sorted(metadata.items())
    }


def _canonicalize_json_value(value: Any) -> JSONValue:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValidationError("metadata values must be finite JSON values")
        return value
    if isinstance(value, Mapping):
        return {
            _require_non_empty_string(key, "metadata key"): _canonicalize_json_value(item)
            for key, item in sorted(value.items())
        }
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return [_canonicalize_json_value(item) for item in value]
    raise ValidationError(f"unsupported metadata value type: {type(value).__name__}")
