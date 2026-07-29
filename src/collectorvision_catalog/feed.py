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
    ValidationError,
    _require_mapping,
    _require_non_empty_string,
    normalize_rfc3339_utc,
)
from .publication import BASE_ASSETS, CatalogVersionManifest, PublishedAsset
from .versioning import validate_public_name

FEED_FILENAME = "catalog-feed-v2.json"
PUBLIC_BASE_URL = "https://hanclinto.github.io/CollectorVisionCatalog/catalog-v2"
_CHECKSUM = re.compile(r"^[0-9a-f]{64}$")


def _exact_keys(payload: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(payload) != expected:
        raise ValidationError(f"{name} fields must be exactly {sorted(expected)}")


def _version(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValidationError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class FileReference:
    url: str
    sha256: str
    size: int

    @property
    def filename(self) -> str:
        return PurePosixPath(unquote(urlparse(self.url).path)).name

    def to_dict(self) -> dict[str, str | int]:
        return {"url": self.url, "sha256": self.sha256, "size": self.size}

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
    manifest: FileReference
    assets: dict[str, FileReference]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "manifest": self.manifest.to_dict(),
            "assets": {key: value.to_dict() for key, value in sorted(self.assets.items())},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SnapshotReference:
        _exact_keys(payload, {"version", "manifest", "assets"}, "feed base")
        return cls(
            version=_version(payload.get("version"), "feed base version"),
            manifest=FileReference.from_dict(
                _require_mapping(payload.get("manifest"), "feed base manifest")
            ),
            assets=_assets_from_dict(
                payload,
                name="feed base",
                allowed=set(BASE_ASSETS),
                required=set(BASE_ASSETS),
            ),
        )


@dataclass(frozen=True)
class DeltaReference:
    from_version: int
    to_version: int
    manifest: FileReference
    assets: dict[str, FileReference]

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_version": self.from_version,
            "to_version": self.to_version,
            "manifest": self.manifest.to_dict(),
            "assets": {key: value.to_dict() for key, value in sorted(self.assets.items())},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DeltaReference:
        _exact_keys(
            payload,
            {"from_version", "to_version", "manifest", "assets"},
            "feed delta",
        )
        assets = _assets_from_dict(
            payload, name="feed delta", allowed=set(BASE_ASSETS), required=set()
        )
        if not assets:
            raise ValidationError("feed delta must contain at least one asset")
        from_version = _version(payload.get("from_version"), "feed delta from_version")
        to_version = _version(payload.get("to_version"), "feed delta to_version")
        if to_version != from_version + 1:
            raise ValidationError("feed delta must advance exactly one version")
        return cls(
            from_version=from_version,
            to_version=to_version,
            manifest=FileReference.from_dict(
                _require_mapping(payload.get("manifest"), "feed delta manifest")
            ),
            assets=assets,
        )


@dataclass(frozen=True)
class CatalogFeedEntry:
    public_name: str
    current_version: int
    source_updated_at: str
    base: SnapshotReference
    deltas: tuple[DeltaReference, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "public_name": self.public_name,
            "current_version": self.current_version,
            "source_updated_at": self.source_updated_at,
            "base": self.base.to_dict(),
            "deltas": [delta.to_dict() for delta in self.deltas],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CatalogFeedEntry:
        _exact_keys(
            payload,
            {"public_name", "current_version", "source_updated_at", "base", "deltas"},
            "catalog feed entry",
        )
        public_name = validate_public_name(payload.get("public_name"))
        current_version = _version(payload.get("current_version"), "feed current_version")
        base = SnapshotReference.from_dict(_require_mapping(payload.get("base"), "feed base"))
        raw_deltas = payload.get("deltas")
        if not isinstance(raw_deltas, list):
            raise ValidationError("feed deltas must be a list")
        deltas = tuple(
            DeltaReference.from_dict(_require_mapping(item, "feed delta")) for item in raw_deltas
        )
        expected = (
            base.version - 1
            if deltas and deltas[0].to_version == base.version
            else base.version
        )
        for delta in deltas:
            if delta.from_version != expected:
                raise ValidationError("feed delta chain is not contiguous")
            expected = delta.to_version
        reached = base.version if not deltas else deltas[-1].to_version
        if reached != current_version:
            raise ValidationError("feed delta chain does not reach current_version")
        entry = cls(
            public_name=public_name,
            current_version=current_version,
            source_updated_at=normalize_rfc3339_utc(
                _require_non_empty_string(
                    payload.get("source_updated_at"), "catalog source_updated_at"
                )
            ),
            base=base,
            deltas=deltas,
        )
        _validate_reference_urls(entry)
        return entry


@dataclass(frozen=True)
class CatalogFeed:
    checked_at: str
    catalogs: dict[str, CatalogFeedEntry]

    def to_dict(self) -> dict[str, Any]:
        return {
            "checked_at": self.checked_at,
            "catalogs": {key: value.to_dict() for key, value in sorted(self.catalogs.items())},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CatalogFeed:
        _exact_keys(payload, {"checked_at", "catalogs"}, "catalog feed")
        catalogs = {
            _require_non_empty_string(key, "catalog feed key"): CatalogFeedEntry.from_dict(
                _require_mapping(value, f"catalog feed {key!r}")
            )
            for key, value in _require_mapping(
                payload.get("catalogs"), "catalog feed catalogs"
            ).items()
        }
        if not catalogs:
            raise ValidationError("catalog feed must contain at least one catalog")
        public_names = [entry.public_name for entry in catalogs.values()]
        if len(public_names) != len(set(public_names)):
            raise ValidationError("catalog feed public names must be unique")
        return cls(
            checked_at=normalize_rfc3339_utc(
                _require_non_empty_string(payload.get("checked_at"), "catalog feed checked_at")
            ),
            catalogs=catalogs,
        )


CatalogHistory = Sequence[tuple[str | Path, CatalogVersionManifest]]


def update_catalog_feed(
    catalog_histories: Mapping[str, CatalogHistory], *, checked_at: str
) -> CatalogFeed:
    """Build the active feed from each catalog's complete ordered manifest history."""
    if not catalog_histories:
        raise ValidationError("catalog feed must contain at least one catalog")
    entries: dict[str, CatalogFeedEntry] = {}
    public_names: set[str] = set()
    for catalog_key, history in sorted(catalog_histories.items()):
        key = _require_non_empty_string(catalog_key, "catalog feed key")
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
            _validate_manifest_files(Path(records[index][0]), manifest)
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
        base_path = Path(records[base_index][0])
        base = SnapshotReference(
            version=base_manifest.version,
            manifest=_file_reference(base_path, public_name, base_manifest.version, base_path.name),
            assets=_asset_references(public_name, base_manifest.version, base_manifest.base or {}),
        )
        delta_indexes = list(range(base_index + 1, len(manifests)))
        if base_manifest.delta is not None:
            delta_indexes.insert(0, base_index)
        deltas = tuple(
            _delta_reference(Path(records[index][0]), manifests[index]) for index in delta_indexes
        )
        current = manifests[-1]
        entries[key] = CatalogFeedEntry.from_dict(
            CatalogFeedEntry(
                public_name=public_name,
                current_version=current.version,
                source_updated_at=current.source_revision.updated_at,
                base=base,
                deltas=deltas,
            ).to_dict()
        )
    return CatalogFeed.from_dict(
        CatalogFeed(checked_at=normalize_rfc3339_utc(checked_at), catalogs=entries).to_dict()
    )


def write_catalog_feed(path: str | Path, feed: CatalogFeed) -> None:
    validated = CatalogFeed.from_dict(feed.to_dict())
    Path(path).write_text(
        json.dumps(validated.to_dict(), indent=2, sort_keys=True) + "\n",
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


def _file_reference(
    path: Path, public_name: str, version: int, relative_path: str
) -> FileReference:
    payload = path.read_bytes()
    return FileReference.from_dict(
        {
            "url": _url(public_name, version, relative_path),
            "sha256": sha256(payload).hexdigest(),
            "size": len(payload),
        }
    )


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


def _delta_reference(path: Path, manifest: CatalogVersionManifest) -> DeltaReference:
    if manifest.delta is None:
        raise ValidationError("catalog history after its selected base must contain deltas")
    return DeltaReference(
        from_version=manifest.delta.from_version,
        to_version=manifest.version,
        manifest=_file_reference(path, manifest.public_name, manifest.version, path.name),
        assets=_asset_references(manifest.public_name, manifest.version, manifest.delta.assets),
    )


def _validate_manifest_files(path: Path, manifest: CatalogVersionManifest) -> None:
    try:
        disk_manifest = CatalogVersionManifest.from_dict(
            _require_mapping(json.loads(path.read_text(encoding="utf-8")), "catalog manifest")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationError(f"cannot read catalog manifest {path}") from error
    if disk_manifest != manifest:
        raise ValidationError("manifest file does not match the supplied manifest")
    assets = [*(manifest.base or {}).values()]
    if manifest.delta is not None:
        assets.extend(manifest.delta.assets.values())
    for asset in assets:
        asset_path = path.parent / asset.path
        try:
            payload = asset_path.read_bytes()
        except OSError as error:
            raise ValidationError(f"cannot read published asset {asset.path!r}") from error
        if len(payload) != asset.size or sha256(payload).hexdigest() != asset.sha256:
            raise ValidationError(f"published asset {asset.path!r} failed integrity validation")


def _validate_reference_urls(entry: CatalogFeedEntry) -> None:
    stages: list[tuple[int, FileReference, Mapping[str, FileReference]]] = [
        (entry.base.version, entry.base.manifest, entry.base.assets),
        *((delta.to_version, delta.manifest, delta.assets) for delta in entry.deltas),
    ]
    for version, manifest, assets in stages:
        expected = f"{PUBLIC_BASE_URL}/{entry.public_name}/version/{version}/"
        if not manifest.url.startswith(expected) or any(
            not reference.url.startswith(expected) for reference in assets.values()
        ):
            raise ValidationError("feed file URL does not match its catalog stage")
