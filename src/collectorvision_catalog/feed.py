from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import (
    CatalogManifest,
    ValidationError,
    _canonical_json_bytes,
    _require_mapping,
    _require_non_empty_string,
)
from .index import CatalogIndex

FEED_FILENAME = "catalog-feed-v2.json"
MAX_DELTA_CHAIN = 4


@dataclass(frozen=True)
class ManifestReference:
    version: str
    manifest_filename: str
    sha256: str

    def to_dict(self) -> dict[str, str]:
        return {
            "version": self.version,
            "manifest_filename": self.manifest_filename,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ManifestReference:
        filename = _require_non_empty_string(
            payload.get("manifest_filename"), "feed manifest_filename"
        )
        if Path(filename).name != filename:
            raise ValidationError("feed manifest_filename must be a flat filename")
        sha = _require_non_empty_string(payload.get("sha256"), "feed sha256")
        if len(sha) != 64:
            raise ValidationError("feed sha256 must contain 64 characters")
        return cls(
            version=_require_non_empty_string(payload.get("version"), "feed version"),
            manifest_filename=filename,
            sha256=sha,
        )


@dataclass(frozen=True)
class DeltaReference:
    from_version: str
    to_version: str
    manifest_filename: str
    sha256: str

    def to_dict(self) -> dict[str, str]:
        return {
            "from": self.from_version,
            "to": self.to_version,
            "manifest_filename": self.manifest_filename,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DeltaReference:
        manifest = ManifestReference.from_dict(
            {
                "version": payload.get("to"),
                "manifest_filename": payload.get("manifest_filename"),
                "sha256": payload.get("sha256"),
            }
        )
        return cls(
            from_version=_require_non_empty_string(payload.get("from"), "feed delta from"),
            to_version=manifest.version,
            manifest_filename=manifest.manifest_filename,
            sha256=manifest.sha256,
        )


@dataclass(frozen=True)
class CatalogFeedEntry:
    base: ManifestReference
    deltas: tuple[DeltaReference, ...]

    @property
    def latest_version(self) -> str:
        return self.base.version if not self.deltas else self.deltas[-1].to_version

    def to_dict(self) -> dict[str, Any]:
        return {
            "base": self.base.to_dict(),
            "deltas": [delta.to_dict() for delta in self.deltas],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CatalogFeedEntry:
        base = ManifestReference.from_dict(_require_mapping(payload.get("base"), "feed base"))
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
        return cls(base=base, deltas=deltas)


@dataclass(frozen=True)
class CatalogFeed:
    schema_version: int
    release_version: str
    catalogs: dict[str, CatalogFeedEntry]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "release_version": self.release_version,
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
            catalogs=catalogs,
        )


def update_catalog_feed(
    *,
    current_index: CatalogIndex,
    current_manifests: Mapping[str, CatalogManifest],
    previous_index: CatalogIndex | None = None,
    previous_feed: CatalogFeed | None = None,
) -> CatalogFeed:
    if set(current_index.catalogs) != set(current_manifests):
        raise ValidationError("current index and manifests contain different catalogs")
    if previous_feed is not None and (
        previous_index is None or previous_feed.release_version != previous_index.release_version
    ):
        raise ValidationError("previous feed and index release versions do not match")
    entries: dict[str, CatalogFeedEntry] = {}
    for key, manifest in sorted(current_manifests.items()):
        if manifest.catalog_key != key or manifest.version != current_index.release_version:
            raise ValidationError("current manifest identity does not match its index")
        current_reference = ManifestReference(
            version=current_index.release_version,
            manifest_filename=current_index.catalogs[key].manifest_filename,
            sha256=current_index.catalogs[key].sha256,
        )
        prior = None if previous_feed is None else previous_feed.catalogs.get(key)
        if prior is None and previous_index is not None and key in previous_index.catalogs:
            previous_entry = previous_index.catalogs[key]
            prior = CatalogFeedEntry(
                base=ManifestReference(
                    version=previous_index.release_version,
                    manifest_filename=previous_entry.manifest_filename,
                    sha256=previous_entry.sha256,
                ),
                deltas=(),
            )
        changed = bool(manifest.delta.operations or manifest.delta.metadata_operations)
        can_append = (
            changed
            and prior is not None
            and manifest.previous_version == prior.latest_version
            and len(prior.deltas) < MAX_DELTA_CHAIN
        )
        if can_append:
            entries[key] = CatalogFeedEntry(
                base=prior.base,
                deltas=(
                    *prior.deltas,
                    DeltaReference(
                        from_version=prior.latest_version,
                        to_version=current_reference.version,
                        manifest_filename=current_reference.manifest_filename,
                        sha256=current_reference.sha256,
                    ),
                ),
            )
        elif not changed and prior is not None:
            entries[key] = prior
        else:
            entries[key] = CatalogFeedEntry(base=current_reference, deltas=())
    return CatalogFeed(
        schema_version=2,
        release_version=current_index.release_version,
        catalogs=entries,
    )


def write_catalog_feed(path: str | Path, feed: CatalogFeed) -> None:
    Path(path).write_bytes(_canonical_json_bytes(feed.to_dict()))


def load_catalog_feed(path: str | Path) -> CatalogFeed:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return CatalogFeed.from_dict(_require_mapping(payload, "catalog feed"))
