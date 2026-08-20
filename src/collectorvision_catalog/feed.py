from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote, unquote, urlparse

from .artifacts import (
    CatalogDescriptor,
    ValidationError,
    _require_mapping,
    _require_non_empty_string,
    normalize_rfc3339_utc,
)
from .publication import (
    CatalogVersionManifest,
    ChangeCounts,
    PublishedAsset,
)
from .versioning import validate_public_name

FEED_FILENAME = "catalog-feed-v2.json"
PUBLIC_BASE_URL = "https://hanclinto.github.io/CollectorVisionCatalog/catalog-v2"
DEFAULT_RETAINED_DELTAS = 30
_CHECKSUM = re.compile(r"^[0-9a-f]{64}$")


def _exact_keys(payload: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(payload) != expected:
        raise ValidationError(f"{name} fields must be exactly {sorted(expected)}")


def _version(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValidationError(f"{name} must be a non-negative integer")
    return value


def _version_key(value: Any) -> int:
    if not isinstance(value, str) or not value.isascii() or not value.isdecimal():
        raise ValidationError("feed update keys must be decimal versions")
    version = int(value)
    if str(version) != value:
        raise ValidationError("feed update keys must use canonical decimal versions")
    return version


def _family_name(value: Any) -> str:
    name = _require_non_empty_string(value, "catalog family")
    if "/" in name:
        raise ValidationError("catalog family must not contain '/'")
    return name


def _local_catalog_key(value: Any) -> str:
    key = _require_non_empty_string(value, "family catalog key")
    if "/" not in key or key.startswith("/") or key.endswith("/"):
        raise ValidationError("family catalog key must contain source/game components")
    return key


def _split_catalog_key(value: Any) -> tuple[str, str]:
    key = _require_non_empty_string(value, "catalog key")
    family, separator, local_key = key.partition("/")
    if not separator:
        raise ValidationError("catalog key must contain family and local components")
    return _family_name(family), _local_catalog_key(local_key)


@dataclass(frozen=True)
class EmbeddingContract:
    model: str
    dimensions: int
    dtype: str
    byte_order: str = "little"
    layout: str = "row-major"

    def to_dict(self) -> dict[str, str | int]:
        return {
            "model": self.model,
            "dimensions": self.dimensions,
            "dtype": self.dtype,
            "byte_order": self.byte_order,
            "layout": self.layout,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> EmbeddingContract:
        _exact_keys(
            payload,
            {"model", "dimensions", "dtype", "byte_order", "layout"},
            "embedding contract",
        )
        if payload.get("dtype") != "float16":
            raise ValidationError("embedding dtype must be 'float16'")
        if payload.get("byte_order") != "little":
            raise ValidationError("embedding byte_order must be 'little'")
        if payload.get("layout") != "row-major":
            raise ValidationError("embedding layout must be 'row-major'")
        dimensions = _version(payload.get("dimensions"), "embedding dimensions")
        if dimensions == 0:
            raise ValidationError("embedding dimensions must be positive")
        return cls(
            model=_require_non_empty_string(payload.get("model"), "embedding model"),
            dimensions=dimensions,
            dtype="float16",
        )


@dataclass(frozen=True)
class FileReference:
    url: str
    sha256: str
    size: int

    @property
    def filename(self) -> str:
        return PurePosixPath(unquote(urlparse(self.url).path)).name

    def to_dict(self) -> dict[str, str | int]:
        return {"url": self.url, "size": self.size, "sha256": self.sha256}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FileReference:
        _exact_keys(payload, {"url", "sha256", "size"}, "feed file reference")
        url = _require_non_empty_string(payload.get("url"), "feed file url")
        parsed = urlparse(url)
        decoded = unquote(parsed.path)
        path = PurePosixPath(decoded)
        root = PurePosixPath(urlparse(PUBLIC_BASE_URL).path)
        if (
            parsed.scheme != "https"
            or parsed.netloc != "hanclinto.github.io"
            or parsed.username is not None
            or parsed.password is not None
            or parsed.port is not None
            or parsed.query
            or parsed.fragment
            or not path.is_absolute()
            or ".." in path.parts
            or "\\" in decoded
            or path == root
            or not path.is_relative_to(root)
        ):
            raise ValidationError(f"feed file url must be under {PUBLIC_BASE_URL}")
        checksum = _require_non_empty_string(payload.get("sha256"), "feed file sha256")
        if _CHECKSUM.fullmatch(checksum) is None:
            raise ValidationError("feed file sha256 must be 64 lowercase hexadecimal characters")
        size = payload.get("size")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValidationError("feed file size must be a non-negative integer")
        return cls(url=url, sha256=checksum, size=size)


def _assets_from_dict(
    payload: Mapping[str, Any], *, name: str, allowed: set[str], required: set[str]
) -> dict[str, FileReference]:
    assets_payload = _require_mapping(payload.get("assets"), f"{name} assets")
    if not required.issubset(assets_payload) or not set(assets_payload).issubset(allowed):
        raise ValidationError(
            f"{name} assets must contain {sorted(required)} and only {sorted(allowed)}"
        )
    return {
        _require_non_empty_string(key, f"{name} asset key"): FileReference.from_dict(
            _require_mapping(value, f"{name} asset {key!r}")
        )
        for key, value in assets_payload.items()
    }


@dataclass(frozen=True)
class SnapshotReference:
    version: int
    rows: int
    source_updated_at: str
    assets: dict[str, FileReference]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "rows": self.rows,
            "source_updated_at": self.source_updated_at,
            "assets": {key: value.to_dict() for key, value in sorted(self.assets.items())},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SnapshotReference:
        _exact_keys(
            payload,
            {"version", "rows", "source_updated_at", "assets"},
            "feed base",
        )
        snapshot = cls(
            version=_version(payload.get("version"), "feed base version"),
            rows=_version(payload.get("rows"), "feed base rows"),
            source_updated_at=normalize_rfc3339_utc(
                _require_non_empty_string(
                    payload.get("source_updated_at"), "feed base source_updated_at"
                )
            ),
            assets=_assets_from_dict(
                payload,
                name="feed base",
                allowed={"records", "embeddings"},
                required={"records", "embeddings"},
            ),
        )
        return snapshot


@dataclass(frozen=True)
class DeltaReference:
    from_version: int
    to_version: int
    rows: ChangeCounts
    recognition_rows: int
    metadata_rows: int
    source_updated_at: str
    assets: dict[str, FileReference]

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_version": self.from_version,
            "to_version": self.to_version,
            "rows": self.rows.to_dict(),
            "recognition_rows": self.recognition_rows,
            "metadata_rows": self.metadata_rows,
            "source_updated_at": self.source_updated_at,
            "assets": {key: value.to_dict() for key, value in sorted(self.assets.items())},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DeltaReference:
        _exact_keys(
            payload,
            {
                "from_version",
                "to_version",
                "rows",
                "recognition_rows",
                "metadata_rows",
                "source_updated_at",
                "assets",
            },
            "feed delta",
        )
        from_version = _version(payload.get("from_version"), "feed delta from_version")
        to_version = _version(payload.get("to_version"), "feed delta to_version")
        if to_version != from_version + 1:
            raise ValidationError("feed delta must advance exactly one version")
        delta = cls(
            from_version=from_version,
            to_version=to_version,
            rows=ChangeCounts.from_dict(
                _require_mapping(payload.get("rows"), "feed delta rows"),
                "feed delta rows",
            ),
            recognition_rows=_version(
                payload.get("recognition_rows"), "feed delta recognition_rows"
            ),
            metadata_rows=_version(payload.get("metadata_rows"), "feed delta metadata_rows"),
            source_updated_at=normalize_rfc3339_utc(
                _require_non_empty_string(
                    payload.get("source_updated_at"), "feed update source_updated_at"
                )
            ),
            assets=_assets_from_dict(
                payload,
                name="feed delta",
                allowed={"records", "embeddings"},
                required={"records"},
            ),
        )
        _validate_delta_rows(delta)
        return delta


@dataclass(frozen=True)
class CatalogFeedEntry:
    public_name: str
    descriptor: CatalogDescriptor
    current_version: int
    rows: int
    source_updated_at: str
    base: SnapshotReference
    updates: dict[int, DeltaReference]

    def to_dict(self) -> dict[str, Any]:
        return {
            "public_name": self.public_name,
            "descriptor": self.descriptor.to_dict(),
            "current_version": self.current_version,
            "rows": self.rows,
            "source_updated_at": self.source_updated_at,
            "base": self.base.to_dict(),
            "updates": {
                str(version): update.to_dict()
                for version, update in sorted(self.updates.items())
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CatalogFeedEntry:
        _exact_keys(
            payload,
            {
                "public_name",
                "descriptor",
                "current_version",
                "rows",
                "source_updated_at",
                "base",
                "updates",
            },
            "catalog feed entry",
        )
        public_name = validate_public_name(payload.get("public_name"))
        current_version = _version(payload.get("current_version"), "feed current_version")
        base = SnapshotReference.from_dict(_require_mapping(payload.get("base"), "feed base"))
        raw_updates = _require_mapping(payload.get("updates"), "feed updates")
        updates: dict[int, DeltaReference] = {}
        for raw_version, raw_update in raw_updates.items():
            version = _version_key(raw_version)
            update = DeltaReference.from_dict(
                _require_mapping(raw_update, f"feed update {raw_version!r}")
            )
            if update.to_version != version:
                raise ValidationError("feed update key must match to_version")
            updates[version] = update
        ordered_updates = tuple(updates[version] for version in sorted(updates))
        expected = ordered_updates[0].from_version if ordered_updates else base.version
        route_start = expected
        for update in ordered_updates:
            if update.from_version != expected:
                raise ValidationError("feed update chain is not contiguous")
            expected = update.to_version
        reached = base.version if not ordered_updates else ordered_updates[-1].to_version
        if route_start > base.version:
            raise ValidationError("feed update chain does not connect its base")
        if reached != current_version:
            raise ValidationError("feed update chain does not reach current_version")
        entry = cls(
            public_name=public_name,
            descriptor=CatalogDescriptor.from_dict(
                _require_mapping(payload.get("descriptor"), "feed catalog descriptor")
            ),
            current_version=current_version,
            rows=_version(payload.get("rows"), "feed catalog rows"),
            source_updated_at=normalize_rfc3339_utc(
                _require_non_empty_string(
                    payload.get("source_updated_at"), "catalog source_updated_at"
                )
            ),
            base=base,
            updates={update.to_version: update for update in ordered_updates},
        )
        _validate_reference_urls(entry)
        expected_rows = entry.base.rows
        for update in entry.updates.values():
            if update.from_version >= entry.base.version:
                expected_rows += update.rows.added - update.rows.deleted
        if entry.rows != expected_rows:
            raise ValidationError("feed catalog rows do not match its base and updates")
        return entry


@dataclass(frozen=True)
class CatalogFamily:
    embedding: EmbeddingContract
    catalogs: dict[str, CatalogFeedEntry]

    def to_dict(self) -> dict[str, Any]:
        return {
            "embedding": self.embedding.to_dict(),
            "catalogs": {key: value.to_dict() for key, value in sorted(self.catalogs.items())},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], family_name: str) -> CatalogFamily:
        _exact_keys(payload, {"embedding", "catalogs"}, f"catalog family {family_name!r}")
        catalogs = {
            _local_catalog_key(key): CatalogFeedEntry.from_dict(
                _require_mapping(value, f"catalog family {family_name!r} catalog {key!r}")
            )
            for key, value in _require_mapping(
                payload.get("catalogs"), f"catalog family {family_name!r} catalogs"
            ).items()
        }
        if not catalogs:
            raise ValidationError("catalog family must contain at least one catalog")
        public_names = [entry.public_name for entry in catalogs.values()]
        if len(public_names) != len(set(public_names)):
            raise ValidationError("catalog family public names must be unique")
        return cls(
            embedding=EmbeddingContract.from_dict(
                _require_mapping(payload.get("embedding"), "embedding contract")
            ),
            catalogs=catalogs,
        )


@dataclass(frozen=True)
class CatalogFeed:
    checked_at: str
    families: dict[str, CatalogFamily]

    def to_dict(self) -> dict[str, Any]:
        return {
            "checked_at": self.checked_at,
            "families": {
                key: value.to_dict() for key, value in sorted(self.families.items())
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CatalogFeed:
        _exact_keys(payload, {"checked_at", "families"}, "catalog feed")
        families = {
            _family_name(key): CatalogFamily.from_dict(
                _require_mapping(value, f"catalog family {key!r}"), key
            )
            for key, value in _require_mapping(
                payload.get("families"), "catalog feed families"
            ).items()
        }
        if not families:
            raise ValidationError("catalog feed must contain at least one family")
        public_names = [
            entry.public_name
            for family in families.values()
            for entry in family.catalogs.values()
        ]
        if len(public_names) != len(set(public_names)):
            raise ValidationError("catalog feed public names must be globally unique")
        return cls(
            checked_at=normalize_rfc3339_utc(
                _require_non_empty_string(payload.get("checked_at"), "catalog feed checked_at")
            ),
            families=families,
        )


CatalogHistory = Sequence[tuple[str | Path, CatalogVersionManifest]]


def update_catalog_feed(
    catalog_histories: Mapping[str, CatalogHistory],
    *,
    checked_at: str,
    retained_deltas: int = DEFAULT_RETAINED_DELTAS,
) -> CatalogFeed:
    """Build the active feed from each catalog's complete ordered manifest history."""
    if not catalog_histories:
        raise ValidationError("catalog feed must contain at least one catalog")
    if retained_deltas <= 0:
        raise ValidationError("retained_deltas must be positive")
    family_entries: dict[str, dict[str, CatalogFeedEntry]] = {}
    family_contracts: dict[str, EmbeddingContract] = {}
    public_names: set[str] = set()
    for catalog_key, history in sorted(catalog_histories.items()):
        key = _require_non_empty_string(catalog_key, "catalog feed key")
        family_name, local_key = _split_catalog_key(key)
        records = list(history)
        if not records:
            raise ValidationError(f"catalog {key!r} has no manifest history")
        manifests = [manifest for _, manifest in records]
        for index, manifest in enumerate(manifests):
            if manifest.catalog_key != key:
                raise ValidationError("manifest catalog_key does not match its feed key")
            if index and manifest.version != manifests[index - 1].version + 1:
                raise ValidationError("catalog manifest history must be ordered and contiguous")
            expected_previous = None if manifest.version == 0 else manifest.version - 1
            if manifest.previous_version != expected_previous:
                raise ValidationError("catalog manifest previous_version is inconsistent")
            if index and manifest.public_name != manifests[0].public_name:
                raise ValidationError("catalog public_name changed within its history")
            _validate_receipt_assets(Path(records[index][0]), manifest)
        public_name = manifests[0].public_name
        if public_name in public_names:
            raise ValidationError("catalog feed public names must be unique")
        public_names.add(public_name)
        base_index = next(
            (index for index in range(len(manifests) - 1, -1, -1) if manifests[index].base),
            None,
        )
        if base_index is None:
            raise ValidationError(f"catalog {key!r} has no base")
        base_manifest = manifests[base_index]
        base = SnapshotReference(
            version=base_manifest.version,
            rows=base_manifest.base.rows,
            source_updated_at=base_manifest.source_revision.updated_at,
            assets=_asset_references(
                public_name, base_manifest.version, base_manifest.base.assets
            ),
        )
        hard_checkpoint_index = max(
            (
                index
                for index, manifest in enumerate(manifests)
                if manifest.base is not None and manifest.delta is None
            ),
            default=-1,
        )
        delta_indexes = [
            index
            for index in range(hard_checkpoint_index + 1, len(manifests))
            if manifests[index].delta is not None
        ][-retained_deltas:]
        updates = tuple(
            _delta_reference(manifests[index]) for index in delta_indexes
        )
        current = manifests[-1]
        entry = CatalogFeedEntry.from_dict(
            CatalogFeedEntry(
                public_name=public_name,
                descriptor=current.descriptor,
                current_version=current.version,
                rows=current.rows,
                source_updated_at=current.source_revision.updated_at,
                base=base,
                updates={update.to_version: update for update in updates},
            ).to_dict()
        )
        contract = EmbeddingContract(
            model=current.embedding_model,
            dimensions=current.dim,
            dtype=current.dtype,
        )
        existing_contract = family_contracts.setdefault(family_name, contract)
        if existing_contract != contract:
            raise ValidationError("catalog family embedding contracts must be identical")
        family_entries.setdefault(family_name, {})[local_key] = entry
    families = {
        family_name: CatalogFamily(
            embedding=family_contracts[family_name],
            catalogs=catalogs,
        )
        for family_name, catalogs in family_entries.items()
    }
    return CatalogFeed.from_dict(
        CatalogFeed(
            checked_at=normalize_rfc3339_utc(checked_at),
            families=families,
        ).to_dict()
    )


def advance_catalog_feed(
    feed: CatalogFeed,
    publications: Mapping[str, tuple[str | Path, CatalogVersionManifest]],
    *,
    checked_at: str,
    retained_deltas: int = DEFAULT_RETAINED_DELTAS,
) -> CatalogFeed:
    """Advance an existing feed with newly published catalog-local versions."""
    if retained_deltas <= 0:
        raise ValidationError("retained_deltas must be positive")
    families = {
        family_name: CatalogFamily(
            embedding=family.embedding,
            catalogs=dict(family.catalogs),
        )
        for family_name, family in feed.families.items()
    }
    for catalog_key, (version_path, manifest) in sorted(publications.items()):
        family_name, local_key = _split_catalog_key(catalog_key)
        family = families.get(family_name)
        if family is None or local_key not in family.catalogs:
            raise ValidationError(f"catalog {catalog_key!r} is not present in the feed")
        current = family.catalogs[local_key]
        if manifest.catalog_key != catalog_key:
            raise ValidationError("manifest catalog_key does not match its feed key")
        if manifest.public_name != current.public_name:
            raise ValidationError("catalog public_name changed from the existing feed")
        if manifest.previous_version != current.current_version:
            raise ValidationError("published catalog does not advance the current feed version")
        if manifest.version != current.current_version + 1:
            raise ValidationError("published catalog version must advance by exactly one")
        contract = EmbeddingContract(
            model=manifest.embedding_model,
            dimensions=manifest.dim,
            dtype=manifest.dtype,
        )
        if contract != family.embedding:
            raise ValidationError("published catalog changed its embedding contract")
        _validate_receipt_assets(Path(version_path), manifest)

        if manifest.base is None:
            base = current.base
        else:
            base = SnapshotReference(
                version=manifest.version,
                rows=manifest.base.rows,
                source_updated_at=manifest.source_revision.updated_at,
                assets=_asset_references(
                    manifest.public_name,
                    manifest.version,
                    manifest.base.assets,
                ),
            )
        updates = (
            {}
            if manifest.base is not None and manifest.delta is None
            else dict(current.updates)
        )
        if manifest.delta is not None:
            update = _delta_reference(manifest)
            updates[update.to_version] = update
        updates = dict(sorted(updates.items())[-retained_deltas:])

        catalogs = dict(family.catalogs)
        catalogs[local_key] = CatalogFeedEntry.from_dict(
            CatalogFeedEntry(
                public_name=manifest.public_name,
                descriptor=manifest.descriptor,
                current_version=manifest.version,
                rows=manifest.rows,
                source_updated_at=manifest.source_revision.updated_at,
                base=base,
                updates=updates,
            ).to_dict()
        )
        families[family_name] = CatalogFamily(
            embedding=family.embedding,
            catalogs=catalogs,
        )
    return CatalogFeed.from_dict(
        CatalogFeed(
            checked_at=normalize_rfc3339_utc(checked_at),
            families=families,
        ).to_dict()
    )


def write_catalog_feed(path: str | Path, feed: CatalogFeed) -> None:
    validated = CatalogFeed.from_dict(feed.to_dict())
    Path(path).write_text(
        json.dumps(validated.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )


def load_catalog_feed(path: str | Path) -> CatalogFeed:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return CatalogFeed.from_dict(_require_mapping(payload, "catalog feed"))


def _url(public_name: str, version: int, relative_path: str) -> str:
    parts = PurePosixPath(relative_path).parts
    if not parts or ".." in parts or PurePosixPath(relative_path).is_absolute():
        raise ValidationError("feed asset path must be safe and relative")
    suffix = "/".join(quote(part, safe="") for part in parts)
    return f"{PUBLIC_BASE_URL}/{quote(public_name, safe='')}/version/{version}/{suffix}"


def _asset_references(
    public_name: str, version: int, assets: Mapping[str, PublishedAsset]
) -> dict[str, FileReference]:
    return {
        name: FileReference.from_dict(
            {
                "url": _url(public_name, version, asset.path),
                "sha256": asset.sha256,
                "size": asset.size,
            }
        )
        for name, asset in sorted(assets.items())
    }


def _delta_reference(manifest: CatalogVersionManifest) -> DeltaReference:
    if manifest.delta is None:
        raise ValidationError("catalog history after its selected base must contain deltas")
    return DeltaReference(
        from_version=manifest.delta.from_version,
        to_version=manifest.version,
        rows=manifest.delta.rows,
        recognition_rows=manifest.delta.recognition_rows,
        metadata_rows=manifest.delta.metadata_rows,
        source_updated_at=manifest.source_revision.updated_at,
        assets=_asset_references(
            manifest.public_name,
            manifest.version,
            manifest.delta.assets,
        ),
    )


def _validate_receipt_assets(path: Path, manifest: CatalogVersionManifest) -> None:
    version_dir = path if path.is_dir() else path.parent
    assets = [*(manifest.base.assets if manifest.base is not None else {}).values()]
    if manifest.delta is not None:
        assets.extend(manifest.delta.assets.values())
    for asset in assets:
        asset_path = version_dir / asset.path
        try:
            payload = asset_path.read_bytes()
        except OSError as error:
            raise ValidationError(f"cannot read published asset {asset.path!r}") from error
        if len(payload) != asset.size or sha256(payload).hexdigest() != asset.sha256:
            raise ValidationError(f"published asset {asset.path!r} failed integrity validation")


def _validate_reference_urls(entry: CatalogFeedEntry) -> None:
    stages: list[tuple[int, Mapping[str, FileReference]]] = [
        (entry.base.version, entry.base.assets),
        *(
            (update.to_version, update.assets)
            for update in entry.updates.values()
        ),
    ]
    for version, assets in stages:
        expected = f"{PUBLIC_BASE_URL}/{entry.public_name}/version/{version}/"
        if any(not reference.url.startswith(expected) for reference in assets.values()):
            raise ValidationError("feed file URL does not match its catalog stage")


def _validate_delta_rows(delta: DeltaReference) -> None:
    if delta.rows.total == 0:
        raise ValidationError("feed update rows must not be empty")
    if not max(delta.recognition_rows, delta.metadata_rows) <= delta.rows.total <= (
        delta.recognition_rows + delta.metadata_rows
    ):
        raise ValidationError("feed update rows must count unique affected catalog rows")
    if "records" not in delta.assets:
        raise ValidationError("feed delta must include a records asset")
    if "embeddings" in delta.assets and delta.recognition_rows == 0:
        raise ValidationError("feed delta embeddings require recognition changes")
