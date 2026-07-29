from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from .artifacts import (
    CatalogManifest,
    ValidationError,
    _canonical_json_bytes,
    _require_mapping,
    _require_non_empty_string,
    max_source_updated_at,
)


@dataclass(frozen=True)
class CatalogIndexEntry:
    manifest_filename: str
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_filename": self.manifest_filename,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CatalogIndexEntry:
        manifest_filename = _require_non_empty_string(
            payload.get("manifest_filename"),
            "catalog index manifest_filename",
        )
        if Path(manifest_filename).name != manifest_filename:
            raise ValidationError("catalog index manifest_filename must be a flat filename")
        return cls(
            manifest_filename=manifest_filename,
            sha256=_require_non_empty_string(payload.get("sha256"), "catalog index sha256"),
        )


@dataclass(frozen=True)
class CatalogIndex:
    schema_version: int
    release_version: str
    source_updated_at: str
    catalogs: dict[str, CatalogIndexEntry]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "release_version": self.release_version,
            "source_updated_at": self.source_updated_at,
            "catalogs": {
                catalog_key: entry.to_dict() for catalog_key, entry in sorted(self.catalogs.items())
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CatalogIndex:
        schema_version = payload.get("schema_version")
        if schema_version != 2:
            raise ValidationError("catalog index schema_version must be 2")
        catalogs = {
            _require_non_empty_string(
                catalog_key, "catalog index key"
            ): CatalogIndexEntry.from_dict(
                _require_mapping(entry_payload, f"catalogs[{catalog_key!r}]")
            )
            for catalog_key, entry_payload in _require_mapping(
                payload.get("catalogs"), "catalogs"
            ).items()
        }
        release_version = _require_non_empty_string(
            payload.get("release_version"),
            "release_version",
        )
        source_updated_at = _require_non_empty_string(
            payload.get("source_updated_at"), "source_updated_at"
        )
        return cls(
            schema_version=schema_version,
            release_version=release_version,
            source_updated_at=source_updated_at,
            catalogs=catalogs,
        )


def write_catalog_index(
    output_path: str | Path,
    release_version: str,
    manifests: Mapping[str, str | Path],
) -> CatalogIndex:
    release_version = _require_non_empty_string(release_version, "release_version")
    entries: dict[str, CatalogIndexEntry] = {}
    source_revisions = []
    for catalog_key, manifest_path in sorted(manifests.items()):
        stable_key = _require_non_empty_string(catalog_key, "catalog index key")
        manifest_file = Path(manifest_path)
        manifest_bytes = manifest_file.read_bytes()
        manifest = CatalogManifest.from_dict(
            _require_mapping(json.loads(manifest_bytes), "manifest")
        )
        if manifest.catalog_key != stable_key:
            raise ValidationError("manifest catalog_key does not match catalog index key")
        filename = manifest_file.name
        if manifest_file.name != filename:
            raise ValidationError("manifest path must resolve to a flat filename")
        entries[stable_key] = CatalogIndexEntry(
            manifest_filename=filename,
            sha256=sha256(manifest_bytes).hexdigest(),
        )
        source_revisions.append(manifest.source_revision)
    if not entries:
        raise ValidationError("catalog index must contain at least one catalog")
    index = CatalogIndex(
        schema_version=2,
        release_version=release_version,
        source_updated_at=max_source_updated_at(source_revisions),
        catalogs=entries,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_json_bytes(index.to_dict()))
    return index


def load_catalog_index(index_path: str | Path) -> CatalogIndex:
    payload = json.loads(Path(index_path).read_text(encoding="utf-8"))
    return CatalogIndex.from_dict(_require_mapping(payload, "catalog index"))
