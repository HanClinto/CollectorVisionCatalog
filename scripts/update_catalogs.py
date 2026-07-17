#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import re
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from PIL import Image

from collectorvision_catalog import (
    CatalogBuild,
    RecognitionRow,
    ValidationError,
    build_catalog,
    load_catalog_build,
    manifest_filename_for_catalog,
    validate_artifacts,
    write_catalog_index,
)
from collectorvision_catalog.artifacts import Embedder, ImageLoader, default_image_loader
from collectorvision_catalog.quality import apply_quality_rules, load_quality_rules
from collectorvision_catalog.sources.scryfall import normalize_scryfall_card
from collectorvision_catalog.sources.tcgcsv import normalize_tcgcsv_product

USER_AGENT = "CollectorVisionCatalog/0.1 (+https://github.com/HanClinto/CollectorVisionCatalog)"
IMAGE_SUFFIXES = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".webp"}
MILO1_MODEL_ID = (
    "collectorvision@9d45a37ebfe40f22ece70507015645de134dc3ec:"
    "milo-1.0.0@sha256:bd13d8d60383c69da04dce261f32e93fdaeaa8fd618fbc991e7385f71b3d45df"
)


class TCGplayerImageUnavailable(ValidationError):
    pass


@dataclass(frozen=True)
class CatalogConfig:
    key: str
    source: dict[str, Any]
    embedding_model: str
    max_changed_rows: int
    enabled: bool
    seed_required: bool


def load_config(path: Path) -> list[CatalogConfig]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValidationError("config schema_version must be 1")
    raw_catalogs = payload.get("catalogs")
    if not isinstance(raw_catalogs, list):
        raise ValidationError("config catalogs must be a list")
    catalogs: list[CatalogConfig] = []
    seen_keys: set[str] = set()
    for index, raw in enumerate(raw_catalogs):
        if not isinstance(raw, dict):
            raise ValidationError(f"config catalogs[{index}] must be an object")
        key = _required_text(raw.get("key"), f"catalogs[{index}].key")
        if key in seen_keys:
            raise ValidationError(f"duplicate catalog key {key!r}")
        seen_keys.add(key)
        source = raw.get("source")
        if not isinstance(source, dict):
            raise ValidationError(f"catalog {key!r} source must be an object")
        _required_text(source.get("type"), f"catalog {key!r} source.type")
        catalogs.append(
            CatalogConfig(
                key=key,
                source=source,
                embedding_model=_required_text(
                    raw.get("embedding_model"), f"catalog {key!r} embedding_model"
                ),
                max_changed_rows=_non_negative_int(
                    raw.get("max_changed_rows", 5000),
                    f"catalog {key!r} max_changed_rows",
                ),
                enabled=bool(raw.get("enabled", False)),
                seed_required=bool(raw.get("seed_required", True)),
            )
        )
    return catalogs


def fetch_scryfall_rows(source: dict[str, Any]) -> list[RecognitionRow]:
    bulk_type = _required_text(source.get("bulk_type"), "scryfall bulk_type")
    languages = {
        _required_text(language, "scryfall language")
        for language in source.get("languages", [])
    }
    bulk_index = _read_json_url("https://api.scryfall.com/bulk-data")
    entries = bulk_index.get("data", [])
    entry = next(
        (
            candidate
            for candidate in entries
            if isinstance(candidate, dict) and candidate.get("type") == bulk_type
        ),
        None,
    )
    if entry is None:
        raise ValidationError(f"Scryfall bulk data does not contain type {bulk_type!r}")
    download_url = entry.get("jsonl_download_uri")
    if not isinstance(download_url, str) or not download_url:
        raise ValidationError(f"Scryfall bulk type {bulk_type!r} has no JSONL download")

    rows: list[RecognitionRow] = []
    request = Request(download_url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
    with urlopen(request, timeout=120) as response:
        with gzip.GzipFile(fileobj=response) as compressed:
            for line_number, raw_line in enumerate(compressed, start=1):
                if not raw_line.strip():
                    continue
                try:
                    card = json.loads(raw_line)
                except json.JSONDecodeError as error:
                    raise ValidationError(
                        f"invalid Scryfall JSONL at line {line_number}: {error}"
                    ) from error
                if source.get("paper_only", False) and "paper" not in card.get("games", []):
                    continue
                if languages and card.get("lang") not in languages:
                    continue
                rows.extend(normalize_scryfall_card(card))
    if not rows:
        raise ValidationError(f"Scryfall bulk type {bulk_type!r} produced no recognition rows")
    return rows


def fetch_tcgcsv_rows(source: dict[str, Any]) -> list[RecognitionRow]:
    category_id = _non_negative_int(source.get("category_id"), "tcgcsv category_id")
    if category_id == 0:
        raise ValidationError("tcgcsv category_id must be positive")
    workers = _non_negative_int(source.get("fetch_workers", 8), "tcgcsv fetch_workers")
    if workers == 0:
        raise ValidationError("tcgcsv fetch_workers must be positive")

    category_payload = _read_json_url("https://tcgcsv.com/tcgplayer/categories")
    category = next(
        (
            item
            for item in _response_results(category_payload, "TCGCSV categories")
            if item.get("categoryId") == category_id
        ),
        None,
    )
    if category is None:
        raise ValidationError(f"TCGCSV does not contain category {category_id}")

    groups_payload = _read_json_url(
        f"https://tcgcsv.com/tcgplayer/{category_id}/groups"
    )
    groups = _response_results(groups_payload, f"TCGCSV category {category_id} groups")

    def fetch_group(group: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        group_id = _non_negative_int(group.get("groupId"), "tcgcsv groupId")
        payload = _read_json_url(
            f"https://tcgcsv.com/tcgplayer/{category_id}/{group_id}/products"
        )
        return group, _response_results(payload, f"TCGCSV group {group_id} products")

    rows: list[RecognitionRow] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for group, products in executor.map(fetch_group, groups):
            for product in products:
                rows.extend(
                    normalize_tcgcsv_product(
                        product,
                        group=group,
                        category=category,
                    )
                )
    if not rows:
        raise ValidationError(f"TCGCSV category {category_id} produced no recognition rows")
    return rows


def build_enabled_catalogs(
    *,
    config_path: Path,
    quality_overrides_path: Path = Path("config/source-quality-overrides.json"),
    previous_dir: Path,
    output_dir: Path,
    version: str,
    allow_full_rebuild: bool = False,
    image_dirs: Sequence[Path] = (),
    cache_root: Path | None = None,
    batch_size: int = 16,
    source_rows_factory: Callable[[dict[str, Any]], list[RecognitionRow]] | None = None,
    embedder_factory: Callable[[str, int], Embedder] | None = None,
    image_loader: ImageLoader | None = None,
) -> dict[str, Any]:
    configs = [catalog for catalog in load_config(config_path) if catalog.enabled]
    if not configs:
        raise ValidationError("config does not enable any catalogs")
    embedder_factory = embedder_factory or create_embedder
    output_dir.mkdir(parents=True, exist_ok=True)
    manifests: dict[str, Path] = {}
    summaries = []
    quality_reports: dict[str, Any] = {}
    quality_rules = load_quality_rules(quality_overrides_path)

    for config in configs:
        source_type = config.source.get("type")
        if source_rows_factory is not None:
            effective_rows_factory = source_rows_factory
        elif source_type == "scryfall":
            effective_rows_factory = fetch_scryfall_rows
        elif source_type == "tcgcsv":
            effective_rows_factory = fetch_tcgcsv_rows
        else:
            raise ValidationError(f"unsupported source type {source_type!r}")
        previous_manifest = previous_dir / manifest_filename_for_catalog(config.key)
        previous: CatalogBuild | None = None
        if previous_manifest.exists():
            previous = load_catalog_build(previous_manifest, asset_dir=previous_dir)
        elif config.seed_required and not allow_full_rebuild:
            raise ValidationError(
                f"catalog {config.key!r} requires a seed release; "
                "run locally with --allow-full-rebuild for the first build"
            )

        rows = effective_rows_factory(config.source)
        quality_result = apply_quality_rules(
            rows,
            source_type=str(source_type),
            rules=quality_rules,
        )
        rows = list(quality_result.rows)
        quality_reports[config.key] = quality_result.report()
        if previous is not None:
            changed_rows = _count_changed_image_rows(rows, previous)
            if changed_rows > config.max_changed_rows:
                raise ValidationError(
                    f"catalog {config.key!r} has {changed_rows:,} image changes, exceeding "
                    f"its safety limit of {config.max_changed_rows:,}; run a reviewed local rebuild"
                )
        if image_loader is not None:
            effective_image_loader = image_loader
        elif cache_root is not None:
            if source_type == "scryfall":
                effective_image_loader = ScryfallImageCache(cache_root, rows)
            elif source_type == "tcgcsv":
                previous_fingerprints = (
                    {}
                    if previous is None
                    else {row.key: row.image_fingerprint for row in previous.rows}
                )
                refresh_urls = (
                    row.image_url
                    for row in rows
                    if previous_fingerprints.get(row.key) != row.image_fingerprint
                )
                effective_image_loader = TCGplayerImageCache(
                    cache_root,
                    rows,
                    refresh_urls=refresh_urls,
                )
            else:
                raise AssertionError(f"unsupported source type {source_type!r}")
        else:
            fallback_loader = (
                tcgplayer_network_image_loader
                if source_type == "tcgcsv"
                else default_image_loader
            )
            effective_image_loader = local_first_image_loader(
                rows,
                image_dirs,
                fallback_loader=fallback_loader,
            )
        build = build_catalog(
            rows,
            embedder=embedder_factory(config.embedding_model, batch_size),
            output_dir=output_dir,
            catalog_key=config.key,
            version=version,
            embedding_model=config.embedding_model,
            previous_build=previous,
            image_loader=effective_image_loader,
            batch_size=batch_size,
        )
        manifest_path = output_dir / manifest_filename_for_catalog(config.key)
        validate_artifacts(
            manifest_path,
            asset_dir=output_dir,
            previous_build=previous,
        )
        manifests[config.key] = manifest_path
        summaries.append(
            {
                "catalog_key": config.key,
                "rows": build.manifest.rows,
                "delta_operations": build.manifest.delta.operations,
                "metadata_delta_operations": build.manifest.delta.metadata_operations,
                "changed": previous is None
                or build.manifest.delta.operations > 0
                or build.manifest.delta.metadata_operations > 0,
                "quality_excluded_rows": len(quality_result.findings),
            }
        )

    index_path = output_dir / "catalog-index-v2.json"
    write_catalog_index(index_path, version, manifests)
    summary = {
        "version": version,
        "changed": any(item["changed"] for item in summaries),
        "catalogs": summaries,
        "quality_excluded_rows": sum(
            report["excluded_rows"] for report in quality_reports.values()
        ),
    }
    (output_dir / "update-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "quality-report.json").write_text(
        json.dumps(
            {"version": version, "catalogs": quality_reports},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def create_embedder(embedding_model: str, batch_size: int) -> Embedder:
    if embedding_model != MILO1_MODEL_ID:
        raise ValidationError(f"unsupported embedding model {embedding_model!r}")
    try:
        from collector_vision import NeuralEmbedder
    except ImportError as error:
        raise RuntimeError(
            "Catalog updates require CollectorVision and exactly one ONNX Runtime backend"
        ) from error
    provider = os.environ.get("COLLECTORVISION_PROVIDER", "auto")
    return NeuralEmbedder(
        family="milo",
        version="1.0.0",
        batch_size=batch_size,
        provider=provider,
    ).embed


def local_first_image_loader(
    rows: Iterable[RecognitionRow],
    image_dirs: Sequence[Path],
    *,
    fallback_loader: ImageLoader = default_image_loader,
) -> ImageLoader:
    roots = [root.resolve() for root in image_dirs]
    if not roots:
        return fallback_loader
    files_by_stem: dict[str, Path] = {}
    for root in roots:
        if not root.is_dir():
            raise ValidationError(f"image cache directory does not exist: {root}")
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                files_by_stem.setdefault(path.stem, path)

    url_to_path: dict[str, Path] = {}
    for row in rows:
        primary = row.primary_id.value
        candidates = (
            [f"{primary}_back", f"{primary}_{row.face_index}", primary]
            if row.face_index > 0
            else [primary, f"{primary}_front", f"{primary}_0"]
        )
        local_path = next(
            (files_by_stem[name] for name in candidates if name in files_by_stem),
            None,
        )
        if local_path is not None:
            url_to_path[row.image_url] = local_path

    def load(image_url: str) -> Image.Image:
        local_path = url_to_path.get(image_url)
        if local_path is None:
            return fallback_loader(image_url)
        with Image.open(local_path) as image:
            loaded = image.convert("RGB")
            loaded.load()
        return loaded

    return load


def tcgplayer_network_image_loader(image_url: str) -> Image.Image:
    payload = _download_tcgplayer_image(image_url)
    with Image.open(io.BytesIO(payload)) as image:
        loaded = image.convert("RGB")
        loaded.load()
    return loaded


class ScryfallImageCache:
    def __init__(self, cache_root: Path, rows: Iterable[RecognitionRow]) -> None:
        self.images_root = _resolve_scryfall_images_root(cache_root)
        self._entries = {
            row.image_url: (
                self.path_for_row(row),
                scryfall_image_revision(row.image_url),
            )
            for row in rows
        }

    def path_for_row(self, row: RecognitionRow) -> Path:
        face = "back" if row.face_index > 0 else "front"
        card_id = row.primary_id.value
        return self.images_root / face / card_id[0] / card_id[1] / f"{card_id}.png"

    def is_current(self, row: RecognitionRow) -> bool:
        path = self.path_for_row(row)
        revision = scryfall_image_revision(row.image_url)
        return path.is_file() and (
            revision == 0 or abs(path.stat().st_mtime - revision) < 1.0
        )

    def __call__(self, image_url: str) -> Image.Image:
        try:
            path, revision = self._entries[image_url]
        except KeyError as error:
            raise ValidationError(
                f"image URL is not part of this Scryfall build: {image_url}"
            ) from error
        if path.is_file() and (
            revision == 0 or abs(path.stat().st_mtime - revision) < 1.0
        ):
            return _open_rgb_image(path)

        request = Request(
            image_url,
            headers={"User-Agent": USER_AGENT, "Accept": "image/*"},
        )
        with urlopen(request, timeout=60) as response:
            payload = response.read()
        with Image.open(io.BytesIO(payload)) as image:
            image.verify()

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        try:
            temporary.write_bytes(payload)
            os.replace(temporary, path)
            if revision:
                os.utime(path, (revision, revision))
        finally:
            temporary.unlink(missing_ok=True)
        return _open_rgb_image(path)


def _download_tcgplayer_image(image_url: str, attempts: int = 6) -> bytes:
    candidate_urls = [image_url]
    if match := re.search(r"_(\d+)_in_", image_url):
        image_number = int(match.group(1))
        candidate_urls.append(
            f"{image_url[: match.start(1)]}{image_number + 1}{image_url[match.end(1) :]}"
        )
    elif "_in_" in image_url:
        candidate_urls.append(image_url.replace("_in_", "_1_in_", 1))
    last_error: HTTPError | URLError | None = None
    for attempt in range(attempts):
        candidate_errors: list[HTTPError | URLError] = []
        for candidate_url in candidate_urls:
            request = Request(
                candidate_url,
                headers={"User-Agent": USER_AGENT, "Accept": "image/*"},
            )
            try:
                with urlopen(request, timeout=60) as response:
                    return response.read()
            except HTTPError as error:
                last_error = error
                candidate_errors.append(error)
                retryable = error.code in {403, 408, 429} or error.code >= 500
                if not retryable:
                    raise ValidationError(
                        f"failed to download TCGplayer image {candidate_url}: HTTP {error.code}"
                    ) from error
            except URLError as error:
                last_error = error
                candidate_errors.append(error)
        if candidate_errors and all(
            isinstance(error, HTTPError) and error.code in {403, 404}
            for error in candidate_errors
        ):
            raise TCGplayerImageUnavailable(
                f"TCGplayer has no available image variant for {image_url}"
            ) from last_error
        time.sleep(min(2**attempt, 30))
    if isinstance(last_error, HTTPError):
        detail = f"HTTP {last_error.code}"
    else:
        detail = str(last_error.reason) if last_error is not None else "unknown error"
    raise ValidationError(
        f"failed to download TCGplayer image variants for {image_url}: {detail}"
    ) from last_error


class TCGplayerImageCache:
    def __init__(
        self,
        cache_root: Path,
        rows: Iterable[RecognitionRow],
        *,
        refresh_urls: Iterable[str] = (),
    ) -> None:
        self.images_root = _resolve_tcgplayer_images_root(cache_root)
        self._entries = {row.image_url: self.path_for_row(row) for row in rows}
        self._refresh_urls = frozenset(refresh_urls)

    def path_for_row(self, row: RecognitionRow) -> Path:
        product_id = row.primary_id.value
        suffix = "" if row.face_index == 0 else f"_{row.face_index}"
        return (
            self.images_root
            / product_id[0]
            / product_id[1]
            / f"{product_id}{suffix}.jpg"
        )

    def is_cached(self, row: RecognitionRow) -> bool:
        return _is_valid_image_file(self.path_for_row(row))

    def is_temporarily_unavailable(self, row: RecognitionRow) -> bool:
        marker = _unavailable_marker(self.path_for_row(row))
        return marker.is_file() and time.time() - marker.stat().st_mtime < 86400

    def __call__(self, image_url: str) -> Image.Image:
        try:
            path = self._entries[image_url]
        except KeyError as error:
            raise ValidationError(
                f"image URL is not part of this TCGplayer build: {image_url}"
            ) from error
        if _is_valid_image_file(path) and image_url not in self._refresh_urls:
            return _open_rgb_image(path)

        try:
            payload = _download_tcgplayer_image(image_url)
        except TCGplayerImageUnavailable:
            marker = _unavailable_marker(path)
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.touch()
            raise
        with Image.open(io.BytesIO(payload)) as image:
            image.verify()

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        try:
            temporary.write_bytes(payload)
            os.replace(temporary, path)
            _unavailable_marker(path).unlink(missing_ok=True)
        finally:
            temporary.unlink(missing_ok=True)
        return _open_rgb_image(path)


def scryfall_image_revision(image_url: str) -> int:
    query = urlparse(image_url).query
    return int(query) if query.isdecimal() else 0


def _resolve_scryfall_images_root(cache_root: Path) -> Path:
    root = cache_root.expanduser().resolve()
    candidates = [
        root / "scryfall" / "images" / "png",
        root / "images" / "png",
        root / "png",
        root,
    ]
    for candidate in candidates:
        if (candidate / "front").is_dir() and (candidate / "back").is_dir():
            return candidate
    raise ValidationError(
        f"could not find Scryfall front/back image cache beneath {cache_root}"
    )


def _resolve_tcgplayer_images_root(cache_root: Path) -> Path:
    root = cache_root.expanduser().resolve()
    candidates = [
        root / "tcgplayer" / "images" / "product",
        root / "images" / "product",
        root / "product",
        root,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise ValidationError(f"could not find TCGplayer image cache beneath {cache_root}")


def _open_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        loaded = image.convert("RGB")
        loaded.load()
    return loaded


def _is_valid_image_file(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with Image.open(path) as image:
            image.verify()
    except (OSError, SyntaxError):
        return False
    return True


def _unavailable_marker(path: Path) -> Path:
    return path.with_suffix(f"{path.suffix}.unavailable")


def _read_json_url(url: str) -> dict[str, Any]:
    request = Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    with urlopen(request, timeout=30) as response:
        payload = json.loads(response.read())
    if not isinstance(payload, dict):
        raise ValidationError(f"expected a JSON object from {url}")
    return payload


def _response_results(payload: dict[str, Any], source_name: str) -> list[dict[str, Any]]:
    if payload.get("success") is not True:
        raise ValidationError(f"{source_name} request failed: {payload.get('errors', [])}")
    results = payload.get("results")
    if not isinstance(results, list) or any(not isinstance(item, dict) for item in results):
        raise ValidationError(f"{source_name} results must be a list of objects")
    return results


def _count_changed_image_rows(
    rows: Sequence[RecognitionRow],
    previous: CatalogBuild,
) -> int:
    previous_fingerprints = {
        row.key: row.image_fingerprint
        for row in previous.rows
    }
    changed_or_added = sum(
        previous_fingerprints.get(row.key) != row.image_fingerprint
        for row in rows
    )
    current_keys = {row.key for row in rows}
    removed = len(set(previous_fingerprints).difference(current_keys))
    return changed_or_added + removed


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{name} must be a non-empty string")
    return value.strip()


def _non_negative_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValidationError(f"{name} must be a non-negative integer")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/catalogs.json"))
    parser.add_argument(
        "--quality-overrides",
        type=Path,
        default=Path("config/source-quality-overrides.json"),
    )
    parser.add_argument("--previous-dir", type=Path, default=Path("release-cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("release"))
    parser.add_argument("--version", required=True)
    parser.add_argument("--allow-full-rebuild", action="store_true")
    parser.add_argument("--image-dir", type=Path, action="append", default=[])
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_enabled_catalogs(
        config_path=args.config,
        quality_overrides_path=args.quality_overrides,
        previous_dir=args.previous_dir,
        output_dir=args.output_dir,
        version=args.version,
        allow_full_rebuild=args.allow_full_rebuild,
        image_dirs=args.image_dir,
        cache_root=args.cache_root,
        batch_size=args.batch_size,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
