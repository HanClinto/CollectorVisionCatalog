from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

from .artifacts import (
    CatalogDescriptor,
    SourceRevision,
    ValidationError,
    _require_mapping,
    _require_non_empty_string,
    normalize_rfc3339_utc,
)
from .feed import EmbeddingContract
from .publication import CatalogVersionManifest, ChangeCounts, PublishedAsset
from .versioning import validate_public_name

AUDIT_FILENAME = "verification-audit.json"
_CHECKSUM = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{7,40}$")
_KEY_COMPONENT = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_INPUT_NAME = re.compile(r"^[a-z0-9]+(?:[_-][a-z0-9]+)*$")


@dataclass(frozen=True)
class ReleaseAudit:
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return self.payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ReleaseAudit:
        _exact_fields(
            payload,
            {"schema_version", "release", "generator", "inputs", "families"},
            "audit",
        )
        if payload.get("schema_version") != 2:
            raise ValidationError("audit schema_version must be 2")
        release = _require_mapping(payload.get("release"), "audit release")
        _exact_fields(release, {"tag", "published_at"}, "audit release")
        _require_non_empty_string(release.get("tag"), "audit release tag")
        published_at = _require_non_empty_string(
            release.get("published_at"), "audit release published_at"
        )
        if normalize_rfc3339_utc(published_at) != published_at:
            raise ValidationError("audit release published_at must be normalized RFC3339 UTC")
        generator = _require_mapping(payload.get("generator"), "audit generator")
        _exact_fields(generator, {"repository", "commit"}, "audit generator")
        repository = _require_non_empty_string(
            generator.get("repository"), "audit generator repository"
        )
        if repository.count("/") != 1 or any(not part for part in repository.split("/")):
            raise ValidationError("audit generator repository must be an owner/repository name")
        commit = _require_non_empty_string(generator.get("commit"), "audit generator commit")
        if _COMMIT.fullmatch(commit) is None:
            raise ValidationError("audit generator commit must be a lowercase Git commit")

        inputs = _require_mapping(payload.get("inputs"), "audit inputs")
        if not inputs:
            raise ValidationError("audit must contain at least one hashed input")
        for name, value in inputs.items():
            _input_name(name)
            input_payload = _require_mapping(value, f"audit input {name!r}")
            _exact_fields(input_payload, {"path", "sha256"}, f"audit input {name!r}")
            _require_non_empty_string(input_payload.get("path"), f"audit input {name!r} path")
            _checksum(input_payload.get("sha256"), f"audit input {name!r} sha256")

        families = _require_mapping(payload.get("families"), "audit families")
        if not families:
            raise ValidationError("audit must contain at least one family")
        filenames: set[str] = set()
        public_names: set[str] = set()
        for family_name, value in families.items():
            _key_component(family_name, "audit family name")
            family = _require_mapping(value, f"audit family {family_name!r}")
            _exact_fields(family, {"embedding", "catalogs"}, f"audit family {family_name!r}")
            EmbeddingContract.from_dict(
                _require_mapping(family.get("embedding"), f"audit family {family_name!r} embedding")
            )
            catalogs = _require_mapping(
                family.get("catalogs"), f"audit family {family_name!r} catalogs"
            )
            if not catalogs:
                raise ValidationError(f"audit family {family_name!r} must contain catalogs")
            for local_key, catalog_value in catalogs.items():
                _local_catalog_key(local_key)
                catalog = _require_mapping(
                    catalog_value, f"audit catalog {family_name}/{local_key}"
                )
                public_name = _validate_catalog_receipt(
                    catalog,
                    family_name=family_name,
                    local_key=local_key,
                    filenames=filenames,
                )
                if public_name in public_names:
                    raise ValidationError(f"duplicate public_name {public_name!r} in audit")
                public_names.add(public_name)
        return cls(payload=dict(payload))


def assemble_catalog_release(
    publications: Mapping[str, tuple[str | Path, CatalogVersionManifest]],
    output_dir: str | Path,
    *,
    tag: str,
    published_at: str,
    repository: str,
    commit: str,
    inputs: Mapping[str, str | Path],
) -> ReleaseAudit:
    if not publications:
        raise ValidationError("release must contain at least one catalog publication")
    output = Path(output_dir)
    if output.exists():
        raise ValidationError(f"output directory already exists: {output}")
    if not output.parent.is_dir():
        raise ValidationError(f"output parent directory does not exist: {output.parent}")

    families: dict[str, dict[str, Any]] = {}
    files: dict[str, Path] = {}
    for catalog_key, (version_root, receipt) in sorted(publications.items()):
        family, separator, local_key = catalog_key.partition("/")
        if not separator:
            raise ValidationError("catalog key must contain an embedding family")
        _key_component(family, "catalog family")
        _local_catalog_key(local_key)
        if receipt.catalog_key != catalog_key:
            raise ValidationError(
                f"publication key {catalog_key!r} does not match receipt catalog_key"
            )
        contract = EmbeddingContract(
            model=receipt.embedding_model,
            dimensions=receipt.dim,
            dtype=receipt.dtype,
        )
        family_payload = families.setdefault(
            family,
            {"embedding": contract.to_dict(), "catalogs": {}},
        )
        if family_payload["embedding"] != contract.to_dict():
            raise ValidationError("release family embedding contracts do not match")
        if local_key in family_payload["catalogs"]:
            raise ValidationError(f"duplicate release catalog {catalog_key!r}")
        root = Path(version_root)
        family_payload["catalogs"][local_key] = _catalog_receipt(receipt, root, files)

    temporary = output.parent / f".{output.name}.assembling-{uuid4().hex}"
    try:
        temporary.mkdir()
        for filename, source in sorted(files.items()):
            shutil.copyfile(source, temporary / filename)
        payload = {
            "schema_version": 2,
            "release": {
                "tag": _require_non_empty_string(tag, "release tag"),
                "published_at": _require_non_empty_string(published_at, "release published_at"),
            },
            "generator": {
                "repository": _require_non_empty_string(repository, "generator repository"),
                "commit": _require_non_empty_string(commit, "generator commit"),
            },
            "inputs": _audit_inputs(inputs),
            "families": families,
        }
        audit = ReleaseAudit.from_dict(payload)
        (temporary / AUDIT_FILENAME).write_text(
            json.dumps(audit.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )
        validate_catalog_release(temporary)
        os.rename(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return audit


def load_release_audit(path: str | Path) -> ReleaseAudit:
    audit_path = Path(path)
    try:
        payload = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationError(f"cannot load release audit {audit_path}: {error}") from error
    return ReleaseAudit.from_dict(_require_mapping(payload, "audit"))


def validate_catalog_release(release_dir: str | Path) -> ReleaseAudit:
    directory = Path(release_dir)
    if not directory.is_dir():
        raise ValidationError(f"release directory does not exist: {directory}")
    release_files = _flat_release_files(directory)
    audit = load_release_audit(directory / AUDIT_FILENAME)
    expected = {AUDIT_FILENAME}
    for family in _require_mapping(audit.payload["families"], "audit families").values():
        catalogs = _require_mapping(
            _require_mapping(family, "audit family").get("catalogs"),
            "audit family catalogs",
        )
        for catalog in catalogs.values():
            for asset in _iter_assets(_require_mapping(catalog, "audit catalog")):
                filename = _require_non_empty_string(asset.get("filename"), "audit filename")
                if filename in expected:
                    raise ValidationError(f"duplicate audit asset filename {filename!r}")
                expected.add(filename)
                path = directory / filename
                try:
                    payload = path.read_bytes()
                except OSError as error:
                    raise ValidationError(
                        f"cannot read audit asset {filename!r}: {error}"
                    ) from error
                if (
                    len(payload) != asset.get("size")
                    or sha256(payload).hexdigest() != asset.get("sha256")
                ):
                    raise ValidationError(f"audit asset {filename!r} failed integrity validation")
    actual = {path.name for path in release_files}
    if actual != expected:
        raise ValidationError("release files do not match audit assets")
    return audit


def _catalog_receipt(
    receipt: CatalogVersionManifest,
    root: Path,
    files: dict[str, Path],
) -> dict[str, Any]:
    return {
        "public_name": receipt.public_name,
        "descriptor": receipt.descriptor.to_dict(),
        "version": receipt.version,
        "previous_version": receipt.previous_version,
        "source_revision": receipt.source_revision.to_dict(),
        "rows": receipt.rows,
        "base": (
            None
            if receipt.base is None
            else {
                "rows": receipt.base.rows,
                "assets": _audit_assets(receipt, root, receipt.base.assets, files),
            }
        ),
        "update": (
            None
            if receipt.delta is None
            else {
                "from_version": receipt.delta.from_version,
                "rows": receipt.delta.rows.to_dict(),
                "recognition_rows": receipt.delta.recognition_rows,
                "metadata_rows": receipt.delta.metadata_rows,
                "assets": _audit_assets(receipt, root, receipt.delta.assets, files),
            }
        ),
    }


def _audit_assets(
    receipt: CatalogVersionManifest,
    root: Path,
    assets: Mapping[str, PublishedAsset],
    files: dict[str, Path],
) -> dict[str, Any]:
    result = {}
    for name, asset in sorted(assets.items()):
        filename = (
            f"{receipt.public_name}.v{receipt.version}."
            f"{asset.path.replace('/', '.')}"
        )
        source = root / asset.path
        payload = source.read_bytes()
        if len(payload) != asset.size or sha256(payload).hexdigest() != asset.sha256:
            raise ValidationError(f"published asset {asset.path!r} failed integrity validation")
        if filename in files:
            raise ValidationError(f"release filename collision: {filename!r}")
        files[filename] = source
        result[name] = {
            "filename": filename,
            "size": asset.size,
            "sha256": asset.sha256,
        }
    return result


def _iter_assets(catalog: Mapping[str, Any]):
    for route_name in ("base", "update"):
        route = catalog.get(route_name)
        if route is None:
            continue
        route_mapping = _require_mapping(route, f"audit catalog {route_name}")
        yield from _require_mapping(
            route_mapping.get("assets"),
            f"audit catalog {route_name} assets",
        ).values()


def _validate_catalog_receipt(
    catalog: Mapping[str, Any],
    *,
    family_name: str,
    local_key: str,
    filenames: set[str],
) -> str:
    label = f"audit catalog {family_name}/{local_key}"
    _exact_fields(
        catalog,
        {
            "public_name",
            "descriptor",
            "version",
            "previous_version",
            "source_revision",
            "rows",
            "base",
            "update",
        },
        label,
    )
    public_name = validate_public_name(catalog.get("public_name"))
    descriptor = _require_mapping(catalog.get("descriptor"), f"{label} descriptor")
    _exact_fields(
        descriptor,
        {"game", "source", "profile", "description", "result_identifier", "recommended"},
        f"{label} descriptor",
    )
    CatalogDescriptor.from_dict(descriptor)
    SourceRevision.from_dict(
        _require_mapping(catalog.get("source_revision"), f"{label} source_revision")
    )
    version = _non_negative_int(catalog.get("version"), f"{label} version")
    previous = catalog.get("previous_version")
    if previous is not None:
        previous = _non_negative_int(previous, f"{label} previous_version")
        if previous != version - 1:
            raise ValidationError(f"{label} previous_version must immediately precede version")
    rows = _positive_int(catalog.get("rows"), f"{label} rows")
    base_payload = catalog.get("base")
    update_payload = catalog.get("update")
    if base_payload is None and update_payload is None:
        raise ValidationError(f"{label} must contain a base or update")
    if version == 0 and (previous is not None or update_payload is not None):
        raise ValidationError(f"{label} version 0 cannot have a previous version or update")
    if version > 0 and previous is None:
        raise ValidationError(f"{label} nonzero version requires previous_version")
    if base_payload is not None:
        _validate_base(
            _require_mapping(base_payload, f"{label} base"),
            label=label,
            public_name=public_name,
            version=version,
            rows=rows,
            filenames=filenames,
        )
    if update_payload is not None:
        if previous is None:
            raise ValidationError(f"{label} update requires previous_version")
        _validate_update(
            _require_mapping(update_payload, f"{label} update"),
            label=label,
            public_name=public_name,
            version=version,
            previous=previous,
            filenames=filenames,
        )
    return public_name


def _validate_base(
    route: Mapping[str, Any],
    *,
    label: str,
    public_name: str,
    version: int,
    rows: int,
    filenames: set[str],
) -> None:
    _exact_fields(route, {"rows", "assets"}, f"{label} base")
    if _positive_int(route.get("rows"), f"{label} base rows") != rows:
        raise ValidationError(f"{label} base rows must match catalog rows")
    assets = _require_mapping(route.get("assets"), f"{label} base assets")
    if set(assets) != {"records", "embeddings"}:
        raise ValidationError(f"{label} base assets must be exactly records and embeddings")
    _validate_assets(
        assets,
        label=f"{label} base",
        prefix=f"{public_name}.v{version}.base",
        filenames=filenames,
    )


def _validate_update(
    route: Mapping[str, Any],
    *,
    label: str,
    public_name: str,
    version: int,
    previous: int,
    filenames: set[str],
) -> None:
    _exact_fields(
        route,
        {"from_version", "rows", "recognition_rows", "metadata_rows", "assets"},
        f"{label} update",
    )
    if _non_negative_int(route.get("from_version"), f"{label} update from_version") != previous:
        raise ValidationError(f"{label} update from_version must match previous_version")
    changes = ChangeCounts.from_dict(
        _require_mapping(route.get("rows"), f"{label} update rows"),
        f"{label} update rows",
    )
    if changes.total == 0:
        raise ValidationError(f"{label} update rows must not be empty")
    recognition_rows = _non_negative_int(
        route.get("recognition_rows"), f"{label} update recognition_rows"
    )
    metadata_rows = _non_negative_int(
        route.get("metadata_rows"), f"{label} update metadata_rows"
    )
    if not max(recognition_rows, metadata_rows) <= changes.total <= (
        recognition_rows + metadata_rows
    ):
        raise ValidationError(f"{label} update rows must count unique affected rows")
    prefix = f"{public_name}.v{version}.delta-from-{previous}"
    assets = _require_mapping(route.get("assets"), f"{label} update assets")
    if not set(assets).issubset({"records", "embeddings"}):
        raise ValidationError(f"{label} update contains unsupported assets")
    if "records" not in assets:
        raise ValidationError(f"{label} update must include a records asset")
    if "embeddings" in assets and recognition_rows == 0:
        raise ValidationError(f"{label} update embeddings require recognition changes")
    _validate_assets(assets, label=f"{label} update", prefix=prefix, filenames=filenames)


def _validate_assets(
    assets: Mapping[str, Any],
    *,
    label: str,
    prefix: str,
    filenames: set[str],
) -> None:
    suffixes = {
        "embeddings": "embeddings.f16.gz",
        "records": "records.jsonl.gz",
    }
    for name, value in assets.items():
        asset = _require_mapping(value, f"{label} asset {name!r}")
        _exact_fields(asset, {"filename", "size", "sha256"}, f"{label} asset {name!r}")
        filename = _flat_filename(asset.get("filename"), f"{label} asset {name!r} filename")
        expected = f"{prefix}.{suffixes[name]}"
        if filename != expected:
            raise ValidationError(f"{label} asset {name!r} filename must be {expected!r}")
        if filename in filenames:
            raise ValidationError(f"duplicate audit asset filename {filename!r}")
        filenames.add(filename)
        _non_negative_int(asset.get("size"), f"{label} asset {name!r} size")
        _checksum(asset.get("sha256"), f"{label} asset {name!r} sha256")


def _audit_inputs(inputs: Mapping[str, str | Path]) -> dict[str, dict[str, str]]:
    if not inputs:
        raise ValidationError("release must contain at least one hashed input")
    result = {}
    for name, value in sorted(inputs.items()):
        _input_name(name)
        path = Path(value)
        try:
            payload = path.read_bytes()
        except OSError as error:
            raise ValidationError(f"cannot read audit input {path}: {error}") from error
        if not path.is_file():
            raise ValidationError(f"audit input must be a regular file: {path}")
        result[name] = {"path": str(path), "sha256": sha256(payload).hexdigest()}
    return result


def _flat_release_files(directory: Path) -> list[Path]:
    files = []
    for path in directory.iterdir():
        if path.is_symlink() or not path.is_file():
            raise ValidationError(f"release assets must be flat regular files: {path.name}")
        filename = _require_non_empty_string(path.name, "release filename")
        if filename in {".", ".."} or "/" in filename or "\\" in filename:
            raise ValidationError("release filename must be a safe flat filename")
        files.append(path)
    return files


def _flat_filename(value: Any, label: str) -> str:
    filename = _require_non_empty_string(value, label)
    if (
        filename in {".", "..", AUDIT_FILENAME}
        or "/" in filename
        or "\\" in filename
        or Path(filename).is_absolute()
    ):
        raise ValidationError(f"{label} must be a safe flat asset filename")
    return filename


def _local_catalog_key(value: Any) -> str:
    key = _require_non_empty_string(value, "audit local catalog key")
    parts = key.split("/")
    if len(parts) < 2 or any(_KEY_COMPONENT.fullmatch(part) is None for part in parts):
        raise ValidationError(
            "audit local catalog key must contain canonical source/game components"
        )
    return key


def _key_component(value: Any, label: str) -> str:
    component = _require_non_empty_string(value, label)
    if _KEY_COMPONENT.fullmatch(component) is None:
        raise ValidationError(f"{label} must be lowercase kebab-case")
    return component


def _input_name(value: Any) -> str:
    name = _require_non_empty_string(value, "audit input name")
    if _INPUT_NAME.fullmatch(name) is None:
        raise ValidationError("audit input name must be lowercase snake_case or kebab-case")
    return name


def _checksum(value: Any, label: str) -> str:
    checksum = _require_non_empty_string(value, label)
    if _CHECKSUM.fullmatch(checksum) is None:
        raise ValidationError(f"{label} must be 64 lowercase hexadecimal characters")
    return checksum


def _non_negative_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValidationError(f"{label} must be a non-negative integer")
    return value


def _positive_int(value: Any, label: str) -> int:
    result = _non_negative_int(value, label)
    if result == 0:
        raise ValidationError(f"{label} must be positive")
    return result


def _exact_fields(payload: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(payload) != expected:
        raise ValidationError(f"{label} fields must be exactly {sorted(expected)}")
