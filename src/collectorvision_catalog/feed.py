from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlparse

from .artifacts import (
    CatalogManifest,
    ValidationError,
    _canonical_json_bytes,
    _require_mapping,
    _require_non_empty_string,
    normalize_rfc3339_utc,
)
from .index import CatalogIndex, CatalogIndexEntry

FEED_FILENAME = "catalog-feed-v2.json"
MAX_DELTA_CHAIN = 4
PUBLIC_BASE_URL = "https://hanclinto.github.io/CollectorVisionCatalog/catalog-v2"
_BASE_ASSETS = ("identifiers", "embeddings", "metadata")
_DELTA_ASSETS = ("identifiers_delta", "embeddings_delta", "metadata_delta")


@dataclass(frozen=True)
class FileReference:
    url: str
    sha256: str
    size: int

    @property
    def filename(self) -> str:
        return Path(unquote(urlparse(self.url).path)).name

    def to_dict(self) -> dict[str, str | int]:
        return {"url": self.url, "sha256": self.sha256, "size": self.size}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FileReference:
        url = _require_non_empty_string(payload.get("url"), "feed file url")
        parsed = urlparse(url)
        filename = Path(unquote(parsed.path)).name
        if (
            parsed.scheme != "https"
            or not parsed.netloc
            or parsed.query
            or parsed.fragment
            or not filename
            or filename in {".", ".."}
        ):
            raise ValidationError("feed file url must be an absolute HTTPS file URL")
        checksum = _require_non_empty_string(payload.get("sha256"), "feed file sha256")
        if len(checksum) != 64:
            raise ValidationError("feed file sha256 must contain 64 characters")
        size = payload.get("size")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValidationError("feed file size must be a non-negative integer")
        return cls(url=url, sha256=checksum, size=size)


@dataclass(frozen=True)
class SnapshotReference:
    version: str
    manifest: FileReference
    assets: dict[str, FileReference]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "manifest": self.manifest.to_dict(),
            "assets": {
                name: reference.to_dict() for name, reference in sorted(self.assets.items())
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SnapshotReference:
        assets = _parse_assets(payload)
        missing = {"identifiers", "embeddings"}.difference(assets)
        if missing:
            raise ValidationError(f"feed base is missing required assets: {sorted(missing)}")
        return cls(
            version=_require_non_empty_string(payload.get("version"), "feed base version"),
            manifest=FileReference.from_dict(
                _require_mapping(payload.get("manifest"), "feed base manifest")
            ),
            assets=assets,
        )


@dataclass(frozen=True)
class DeltaReference:
    from_version: str
    to_version: str
    manifest: FileReference
    assets: dict[str, FileReference]

    def to_dict(self) -> dict[str, Any]:
        return {
            "from": self.from_version,
            "to": self.to_version,
            "manifest": self.manifest.to_dict(),
            "assets": {
                name: reference.to_dict() for name, reference in sorted(self.assets.items())
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DeltaReference:
        assets = _parse_assets(payload)
        if not assets:
            raise ValidationError("feed delta must contain at least one asset")
        return cls(
            from_version=_require_non_empty_string(payload.get("from"), "feed delta from"),
            to_version=_require_non_empty_string(payload.get("to"), "feed delta to"),
            manifest=FileReference.from_dict(
                _require_mapping(payload.get("manifest"), "feed delta manifest")
            ),
            assets=assets,
        )


@dataclass(frozen=True)
class CatalogFeedEntry:
    source_updated_at: str
    base: SnapshotReference
    deltas: tuple[DeltaReference, ...]

    @property
    def latest_version(self) -> str:
        return self.base.version if not self.deltas else self.deltas[-1].to_version

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_updated_at": self.source_updated_at,
            "base": self.base.to_dict(),
            "deltas": [delta.to_dict() for delta in self.deltas],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CatalogFeedEntry:
        base = SnapshotReference.from_dict(_require_mapping(payload.get("base"), "feed base"))
        raw_deltas = payload.get("deltas")
        if not isinstance(raw_deltas, list):
            raise ValidationError("feed deltas must be a list")
        deltas = tuple(
            DeltaReference.from_dict(_require_mapping(value, "feed delta")) for value in raw_deltas
        )
        expected = base.version
        for delta in deltas:
            if delta.from_version != expected:
                raise ValidationError("feed delta chain is not contiguous")
            if delta.to_version == delta.from_version:
                raise ValidationError("feed delta must advance to another version")
            expected = delta.to_version
        if len(deltas) > MAX_DELTA_CHAIN:
            raise ValidationError("feed delta chain exceeds the supported maximum")
        return cls(
            source_updated_at=normalize_rfc3339_utc(
                _require_non_empty_string(
                    payload.get("source_updated_at"),
                    "catalog feed entry source_updated_at",
                )
            ),
            base=base,
            deltas=deltas,
        )


@dataclass(frozen=True)
class CatalogFeed:
    schema_version: int
    release_version: str
    checked_at: str
    source_updated_at: str
    catalogs: dict[str, CatalogFeedEntry]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "release_version": self.release_version,
            "checked_at": self.checked_at,
            "source_updated_at": self.source_updated_at,
            "catalogs": {key: entry.to_dict() for key, entry in sorted(self.catalogs.items())},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CatalogFeed:
        if payload.get("schema_version") != 2:
            raise ValidationError("catalog feed schema_version must be 2")
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
        return cls(
            schema_version=2,
            release_version=_require_non_empty_string(
                payload.get("release_version"), "catalog feed release_version"
            ),
            checked_at=normalize_rfc3339_utc(
                _require_non_empty_string(payload.get("checked_at"), "catalog feed checked_at")
            ),
            source_updated_at=normalize_rfc3339_utc(
                _require_non_empty_string(
                    payload.get("source_updated_at"), "catalog feed source_updated_at"
                )
            ),
            catalogs=catalogs,
        )


def update_catalog_feed(
    *,
    current_index: CatalogIndex,
    current_manifests: Mapping[str, CatalogManifest],
    checked_at: str,
    previous_index: CatalogIndex | None = None,
    previous_manifests: Mapping[str, CatalogManifest] | None = None,
    previous_feed: CatalogFeed | None = None,
) -> CatalogFeed:
    if set(current_index.catalogs) != set(current_manifests):
        raise ValidationError("current index and manifests contain different catalogs")
    if previous_feed is not None and previous_index is None:
        raise ValidationError("previous feed requires a previous index")
    entries: dict[str, CatalogFeedEntry] = {}
    feed_changed = False
    for key, manifest in sorted(current_manifests.items()):
        if manifest.catalog_key != key or manifest.version != current_index.release_version:
            raise ValidationError("current manifest identity does not match its index")
        current_base = _snapshot_reference(
            current_index.release_version,
            current_index.catalogs[key],
            manifest,
        )
        prior = None if previous_feed is None else previous_feed.catalogs.get(key)
        if (
            prior is None
            and previous_index is not None
            and previous_manifests is not None
            and key in previous_index.catalogs
            and key in previous_manifests
        ):
            prior = CatalogFeedEntry(
                source_updated_at=previous_manifests[key].source_revision.updated_at,
                base=_snapshot_reference(
                    previous_index.release_version,
                    previous_index.catalogs[key],
                    previous_manifests[key],
                ),
                deltas=(),
            )
        changed = prior is None or bool(
            manifest.delta.operations or manifest.delta.metadata_operations
        )
        can_append = (
            changed
            and prior is not None
            and manifest.previous_version == prior.latest_version
            and len(prior.deltas) < MAX_DELTA_CHAIN
        )
        if can_append:
            feed_changed = True
            entries[key] = CatalogFeedEntry(
                source_updated_at=manifest.source_revision.updated_at,
                base=prior.base,
                deltas=(
                    *prior.deltas,
                    _delta_reference(
                        prior.latest_version,
                        current_index.release_version,
                        current_index.catalogs[key],
                        manifest,
                    ),
                ),
            )
        elif not changed and prior is not None:
            entries[key] = CatalogFeedEntry(
                source_updated_at=manifest.source_revision.updated_at,
                base=prior.base,
                deltas=prior.deltas,
            )
        else:
            feed_changed = True
            entries[key] = CatalogFeedEntry(
                source_updated_at=manifest.source_revision.updated_at,
                base=current_base,
                deltas=(),
            )
    release_version = current_index.release_version
    if not feed_changed and previous_feed is not None:
        release_version = previous_feed.release_version
    elif not feed_changed and previous_index is not None:
        release_version = previous_index.release_version
    return CatalogFeed(
        schema_version=2,
        release_version=release_version,
        checked_at=normalize_rfc3339_utc(checked_at),
        source_updated_at=current_index.source_updated_at,
        catalogs=entries,
    )


def write_catalog_feed(path: str | Path, feed: CatalogFeed) -> None:
    Path(path).write_bytes(_canonical_json_bytes(feed.to_dict()))


def load_catalog_feed(path: str | Path) -> CatalogFeed:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return CatalogFeed.from_dict(_require_mapping(payload, "catalog feed"))


def _snapshot_reference(
    version: str,
    index_entry: CatalogIndexEntry,
    manifest: CatalogManifest,
) -> SnapshotReference:
    return SnapshotReference(
        version=version,
        manifest=_manifest_reference(version, index_entry, manifest),
        assets=_asset_references(version, manifest, _BASE_ASSETS),
    )


def _delta_reference(
    from_version: str,
    to_version: str,
    index_entry: CatalogIndexEntry,
    manifest: CatalogManifest,
) -> DeltaReference:
    return DeltaReference(
        from_version=from_version,
        to_version=to_version,
        manifest=_manifest_reference(to_version, index_entry, manifest),
        assets=_asset_references(to_version, manifest, _DELTA_ASSETS),
    )


def _manifest_reference(
    version: str,
    index_entry: CatalogIndexEntry,
    manifest: CatalogManifest,
) -> FileReference:
    payload = _canonical_json_bytes(manifest.to_dict())
    if sha256(payload).hexdigest() != index_entry.sha256:
        raise ValidationError("manifest checksum does not match its index entry")
    return FileReference(
        url=_asset_url(version, index_entry.manifest_filename),
        sha256=index_entry.sha256,
        size=len(payload),
    )


def _asset_references(
    version: str,
    manifest: CatalogManifest,
    names: tuple[str, ...],
) -> dict[str, FileReference]:
    return {
        name: FileReference(
            url=_asset_url(version, manifest.assets[name].filename),
            sha256=manifest.assets[name].sha256,
            size=manifest.assets[name].size,
        )
        for name in names
        if name in manifest.assets
    }


def _asset_url(version: str, filename: str) -> str:
    return f"{PUBLIC_BASE_URL}/{quote(version, safe='')}/{quote(filename, safe='')}"


def _parse_assets(payload: Mapping[str, Any]) -> dict[str, FileReference]:
    return {
        _require_non_empty_string(name, "feed asset name"): FileReference.from_dict(
            _require_mapping(value, f"feed asset {name!r}")
        )
        for name, value in _require_mapping(payload.get("assets"), "feed assets").items()
    }
