from __future__ import annotations

from collections.abc import Callable, Mapping
from hashlib import sha256
from typing import Any

from ..artifacts import RecognitionRow, ValidationError


def normalize_tcgcsv_product(
    product: Mapping[str, Any],
    group: Mapping[str, Any] | None = None,
    category: Mapping[str, Any] | None = None,
    *,
    predicate: Callable[[Mapping[str, Any]], bool] | None = None,
) -> list[RecognitionRow]:
    predicate = predicate or is_probable_card_product
    if not predicate(product):
        return []
    raw_product_id = _lookup(product, "productId", "productID", "product_id")
    if raw_product_id is None:
        raise ValidationError("tcgcsv productId must be present")
    product_id = _require_positive_decimal(raw_product_id, "productId")
    product_name = _require_string(_lookup(product, "name", "productName"), "name")
    image_count_raw = _lookup(product, "imageCount", "image_count")
    image_count = int(image_count_raw) if image_count_raw is not None else 1
    if image_count == 0:
        return []
    if image_count < 0:
        raise ValidationError("tcgcsv imageCount must not be negative")
    extended_data = extract_extended_data(product)
    identifiers: dict[str, str] = {}
    if group is not None:
        group_id = _lookup(group, "groupId", "groupID", "group_id")
        if group_id is not None:
            identifiers["tcgplayer_group"] = str(group_id)
    if category is not None:
        category_id = _lookup(category, "categoryId", "categoryID", "category_id")
        if category_id is not None:
            identifiers["tcgplayer_category"] = str(category_id)
    metadata: dict[str, Any] = {"name": product_name}
    if modified_on := _lookup(product, "modifiedOn", "modified_on"):
        metadata["modified_on"] = str(modified_on)
    if group is not None and (group_name := _lookup(group, "name", "groupName")) is not None:
        metadata["set"] = str(group_name)
    if (
        category is not None
        and (category_name := _lookup(category, "name", "categoryName")) is not None
    ):
        metadata["category"] = str(category_name)
    if collector_number := extended_data.get("Number"):
        metadata["collector_number"] = collector_number
    if rarity := extended_data.get("Rarity"):
        metadata["rarity"] = rarity
    if language := extended_data.get("Language"):
        metadata["language"] = language
    rows: list[RecognitionRow] = []
    for index, image_url in enumerate(
        build_tcgplayer_image_urls(product_id, image_count=image_count)
    ):
        rows.append(
            RecognitionRow(
                provider="tcgplayer",
                id=product_id,
                identifiers=dict(sorted(identifiers.items())),
                image_url=image_url,
                image_fingerprint=_fingerprint(
                    f"{image_url}|{_lookup(product, 'modifiedOn', 'modified_on') or ''}"
                ),
                face_index=index,
                metadata=dict(metadata),
            )
        )
    return rows


def is_probable_card_product(product: Mapping[str, Any]) -> bool:
    extended_data = extract_extended_data(product)
    return bool(extended_data.get("Number") or extended_data.get("Rarity"))


def build_tcgplayer_image_urls(
    product_id: str | int,
    image_count: int,
    *,
    size: str = "1000x1000",
) -> list[str]:
    if image_count <= 0:
        raise ValidationError("image_count must be positive")
    product_id_text = _require_positive_decimal(product_id, "product_id")
    urls: list[str] = []
    for index in range(image_count):
        suffix = "" if index == 0 else f"_{index}"
        urls.append(
            f"https://tcgplayer-cdn.tcgplayer.com/product/{product_id_text}{suffix}_in_{size}.jpg"
        )
    return urls


def extract_extended_data(product: Mapping[str, Any]) -> dict[str, str]:
    raw = _lookup(product, "extendedData", "extended_data")
    if raw is None:
        return {}
    if isinstance(raw, Mapping):
        return {
            _require_string(str(key), "extendedData key"): _require_string(
                str(value),
                f"extendedData[{key!r}]",
            )
            for key, value in raw.items()
            if value not in (None, "")
        }
    if not isinstance(raw, list):
        raise ValidationError("tcgcsv extendedData must be a list or mapping")
    extracted: dict[str, str] = {}
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ValidationError(f"tcgcsv extendedData[{index}] must be a mapping")
        name = _lookup(item, "name", "Name")
        value = _lookup(item, "value", "Value")
        if name in (None, "") or value in (None, ""):
            continue
        extracted[_require_string(str(name), f"extendedData[{index}].name")] = _require_string(
            str(value),
            f"extendedData[{index}].value",
        )
    return extracted


def _lookup(payload: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in payload:
            return payload[name]
    return None


def _fingerprint(image_url: str) -> str:
    return sha256(image_url.encode("utf-8")).hexdigest()


def _require_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"tcgcsv {name} must be a non-empty string")
    return value.strip()


def _require_positive_decimal(value: Any, name: str) -> str:
    text = str(value)
    if not text.isascii() or not text.isdecimal() or int(text) <= 0:
        raise ValidationError(f"tcgcsv {name} must be a positive decimal integer")
    return str(int(text))
