from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

from .artifacts import (
    CatalogManifest,
    ValidationError,
    _canonical_json_bytes,
    _require_mapping,
    _require_non_empty_string,
    max_source_updated_at,
    validate_artifacts,
)
from .index import CatalogIndex, CatalogIndexEntry, load_catalog_index

INDEX_FILENAME = "catalog-index-v2.json"
QUALITY_FILENAME = "quality-report.json"
SEED_SUMMARY_FILENAME = "seed-summary.json"
CHECKSUM_FILENAME = "SHA256SUMS"
_ASSEMBLY_FILENAMES = {INDEX_FILENAME, QUALITY_FILENAME, SEED_SUMMARY_FILENAME, CHECKSUM_FILENAME}
_BETA_VERSION = re.compile(r"^catalog-v2-beta\.\d+-(\d{4}-\d{2}-\d{2})$")


def assemble_seed_release(
    input_dirs: Sequence[str | Path],
    output_dir: str | Path,
    release_version: str,
) -> CatalogIndex:
    """Validate and atomically combine independent seed builds."""
    version = _require_non_empty_string(release_version, "release_version")
    sources = [Path(path) for path in input_dirs]
    if not sources:
        raise ValidationError("at least one input directory is required")
    output = Path(output_dir)
    if output.exists():
        raise ValidationError(f"output directory already exists: {output}")
    if not output.parent.is_dir():
        raise ValidationError(f"output parent directory does not exist: {output.parent}")

    entries: dict[str, CatalogIndexEntry] = {}
    source_files: dict[str, Path] = {}
    quality_catalogs: dict[str, Any] = {}
    summaries: list[dict[str, Any]] = []
    for source in sources:
        index, manifests = _validate_indexed_release(source, expected_version=version)
        _verify_checksums_if_present(source)
        for manifest_path, manifest in manifests.values():
            if (
                manifest.previous_version is not None
                or manifest.delta.base_version is not None
                or manifest.delta.requires_exact_base
                or manifest.delta.operations
                or manifest.delta.metadata_operations
            ):
                raise ValidationError(
                    f"seed manifest has non-empty delta semantics: {manifest_path}"
                )
        quality = _load_versioned_mapping(source / QUALITY_FILENAME, version, "quality report")
        quality_entries = _require_mapping(quality.get("catalogs"), "quality report catalogs")
        summary = _load_versioned_mapping(source / SEED_SUMMARY_FILENAME, version, "seed summary")
        catalog_keys = sorted(index.catalogs)
        if set(quality_entries) != set(catalog_keys):
            raise ValidationError(
                f"quality report catalog keys do not match index in {source}"
            )
        summaries.append({"catalog_keys": catalog_keys, "summary": summary})

        for catalog_key, entry in index.catalogs.items():
            if catalog_key in entries:
                raise ValidationError(f"duplicate catalog key {catalog_key!r}")
            entries[catalog_key] = entry
            quality_catalogs[catalog_key] = quality_entries[catalog_key]
            manifest_path, manifest = manifests[catalog_key]
            _claim_file(source_files, entry.manifest_filename, manifest_path)
            for asset in manifest.assets.values():
                _claim_file(source_files, asset.filename, source / asset.filename)

    summaries.sort(key=lambda item: item["catalog_keys"])
    combined_index = CatalogIndex(
        schema_version=2,
        release_version=version,
        source_updated_at=max_source_updated_at(
            entry.source_revision for entry in entries.values()
        ),
        catalogs=entries,
    )
    _validate_beta_version_date(version, combined_index.source_updated_at)
    temporary = output.parent / f".{output.name}.assembling-{uuid4().hex}"
    try:
        temporary.mkdir()
        for filename, source_path in sorted(source_files.items()):
            shutil.copyfile(source_path, temporary / filename)
        (temporary / INDEX_FILENAME).write_bytes(_canonical_json_bytes(combined_index.to_dict()))
        (temporary / QUALITY_FILENAME).write_bytes(
            _canonical_json_bytes({"version": version, "catalogs": quality_catalogs})
        )
        (temporary / SEED_SUMMARY_FILENAME).write_bytes(
            _canonical_json_bytes(
                {
                    "version": version,
                    "source_updated_at": combined_index.source_updated_at,
                    "catalog_keys": sorted(entries),
                    "inputs": summaries,
                }
            )
        )
        validate_release(temporary, expected_version=version)
        write_checksums(temporary)
        if output.exists():
            raise ValidationError(f"output directory already exists: {output}")
        os.rename(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return combined_index


def validate_release(
    release_dir: str | Path,
    *,
    expected_version: str | None = None,
    verify_checksums: bool = True,
) -> CatalogIndex:
    """Validate a complete flat release and verify SHA256SUMS when it exists."""
    directory = Path(release_dir)
    index, _ = _validate_indexed_release(directory, expected_version=expected_version)
    _flat_release_files(directory)
    checksum_path = directory / CHECKSUM_FILENAME
    if verify_checksums and checksum_path.exists():
        _verify_checksums_if_present(directory)
    return index


def write_checksums(release_dir: str | Path) -> Path:
    """Validate a release and write deterministic checksums for every release file."""
    directory = Path(release_dir)
    validate_release(directory, verify_checksums=False)
    checksum_path = directory / CHECKSUM_FILENAME
    temporary = directory / f".{CHECKSUM_FILENAME}.{uuid4().hex}.tmp"
    try:
        temporary.write_bytes(_checksum_bytes(directory))
        os.replace(temporary, checksum_path)
    finally:
        temporary.unlink(missing_ok=True)
    return checksum_path


def _validate_indexed_release(
    directory: Path,
    *,
    expected_version: str | None,
) -> tuple[CatalogIndex, dict[str, tuple[Path, CatalogManifest]]]:
    if not directory.is_dir():
        raise ValidationError(f"release directory does not exist: {directory}")
    index_path = directory / INDEX_FILENAME
    try:
        index = load_catalog_index(index_path)
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationError(f"cannot load {index_path}: {error}") from error
    if expected_version is not None and index.release_version != expected_version:
        raise ValidationError(
            f"release version {index.release_version!r} does not match expected "
            f"{expected_version!r}"
        )
    if not index.catalogs:
        raise ValidationError("catalog index must contain at least one catalog")
    _validate_beta_version_date(index.release_version, index.source_updated_at)

    manifests: dict[str, tuple[Path, CatalogManifest]] = {}
    claimed = set(_ASSEMBLY_FILENAMES)
    for catalog_key, entry in index.catalogs.items():
        _require_flat_filename(entry.manifest_filename, "manifest filename")
        if entry.manifest_filename in claimed:
            raise ValidationError(f"filename collision: {entry.manifest_filename!r}")
        claimed.add(entry.manifest_filename)
        manifest_path = directory / entry.manifest_filename
        try:
            manifest_bytes = manifest_path.read_bytes()
        except OSError as error:
            raise ValidationError(f"cannot read manifest {manifest_path}: {error}") from error
        if sha256(manifest_bytes).hexdigest() != entry.sha256:
            raise ValidationError(f"manifest checksum mismatch for {entry.manifest_filename}")
        try:
            build = validate_artifacts(manifest_path, asset_dir=directory)
        except (OSError, json.JSONDecodeError) as error:
            raise ValidationError(f"cannot validate manifest {manifest_path}: {error}") from error
        manifest = build.manifest
        if manifest.catalog_key != catalog_key:
            raise ValidationError("index key does not match manifest catalog_key")
        if manifest.descriptor != entry.descriptor:
            raise ValidationError("index descriptor does not match manifest descriptor")
        if manifest.source_revision != entry.source_revision:
            raise ValidationError("index source revision does not match manifest source revision")
        if manifest.version != index.release_version:
            raise ValidationError("manifest version does not match index release_version")
        for asset in manifest.assets.values():
            _require_flat_filename(asset.filename, "asset filename")
            if asset.filename in claimed:
                raise ValidationError(f"filename collision: {asset.filename!r}")
            claimed.add(asset.filename)
        manifests[catalog_key] = (manifest_path, manifest)
    return index, manifests


def _validate_beta_version_date(version: str, source_updated_at: str) -> None:
    match = _BETA_VERSION.fullmatch(version)
    if match is not None and match.group(1) != source_updated_at[:10]:
        raise ValidationError(
            "beta release date suffix must match index source_updated_at UTC date"
        )


def _load_versioned_mapping(path: Path, version: str, label: str) -> Mapping[str, Any]:
    try:
        payload = _require_mapping(json.loads(path.read_text(encoding="utf-8")), label)
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationError(f"cannot load {label} {path}: {error}") from error
    summary_version = payload.get("version")
    if summary_version != version:
        raise ValidationError(f"{label} version does not match expected release version")
    return payload


def _claim_file(files: dict[str, Path], filename: str, source: Path) -> None:
    _require_flat_filename(filename, "release filename")
    if filename in _ASSEMBLY_FILENAMES or filename in files:
        raise ValidationError(f"filename collision: {filename!r}")
    files[filename] = source


def _require_flat_filename(filename: str, label: str) -> None:
    if (
        not filename
        or filename in {".", ".."}
        or "/" in filename
        or "\\" in filename
        or Path(filename).is_absolute()
    ):
        raise ValidationError(f"{label} must be a flat filename")


def _flat_release_files(directory: Path) -> list[Path]:
    files: list[Path] = []
    for path in directory.iterdir():
        if path.is_symlink() or not path.is_file():
            raise ValidationError(f"release assets must be flat regular files: {path.name}")
        _require_flat_filename(path.name, "release filename")
        files.append(path)
    return sorted(files, key=lambda path: path.name)


def _checksum_bytes(directory: Path) -> bytes:
    lines = []
    for path in _flat_release_files(directory):
        if path.name == CHECKSUM_FILENAME:
            continue
        lines.append(f"{sha256(path.read_bytes()).hexdigest()}  {path.name}\n")
    return "".join(lines).encode()


def _verify_checksums_if_present(directory: Path) -> None:
    checksum_path = directory / CHECKSUM_FILENAME
    if checksum_path.exists() and checksum_path.read_bytes() != _checksum_bytes(directory):
        raise ValidationError(f"{CHECKSUM_FILENAME} is stale or malformed")
