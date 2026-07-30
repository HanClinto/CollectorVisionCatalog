from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
from typing import Any
from uuid import UUID

from ..artifacts import RecognitionRow, ValidationError

_IMAGE_PREFERENCE = ("png", "large", "normal")


def normalize_scryfall_card(card: Mapping[str, Any]) -> list[RecognitionRow]:
    card_id = _require_uuid(card.get("id"), "id")
    _require_string(card.get("name"), "name")
    identifiers: dict[str, str] = {}
    if (oracle_id := card.get("oracle_id")) not in (None, ""):
        identifiers["scryfall_oracle"] = _require_uuid(oracle_id, "oracle_id")
    for field_name, source_name in (
        ("tcgplayer_product", "tcgplayer_id"),
        ("tcgplayer_etched_product", "tcgplayer_etched_id"),
    ):
        if (value := card.get(source_name)) not in (None, ""):
            identifiers[field_name] = _require_positive_decimal(value, source_name)
    metadata = {
        field: value
        for field in (
            "name",
            "set",
            "set_name",
            "lang",
            "collector_number",
            "rarity",
            "finishes",
        )
        if (value := card.get(field)) is not None
    }
    rows: list[RecognitionRow] = []
    card_faces = card.get("card_faces") or []
    if isinstance(card_faces, list) and card_faces:
        for index, face_payload in enumerate(card_faces):
            if not isinstance(face_payload, Mapping):
                raise ValidationError("card_faces entries must be mappings")
            image_url = _preferred_image_url(face_payload.get("image_uris"))
            if image_url is None:
                continue
            rows.append(
                RecognitionRow(
                    provider="scryfall",
                    id=card_id,
                    identifiers=dict(sorted(identifiers.items())),
                    image_url=image_url,
                    image_fingerprint=_fingerprint(image_url),
                    face_index=index,
                    metadata=dict(metadata),
                )
            )
        if rows:
            return rows
    image_url = _preferred_image_url(card.get("image_uris"))
    if image_url is None:
        return []
    return [
        RecognitionRow(
            provider="scryfall",
            id=card_id,
            identifiers=dict(sorted(identifiers.items())),
            image_url=image_url,
            image_fingerprint=_fingerprint(image_url),
            metadata=metadata,
        )
    ]


def _preferred_image_url(image_uris: Any) -> str | None:
    if not isinstance(image_uris, Mapping):
        return None
    for field in _IMAGE_PREFERENCE:
        value = image_uris.get(field)
        if isinstance(value, str) and value:
            return value
    return None


def _fingerprint(image_url: str) -> str:
    return sha256(image_url.encode("utf-8")).hexdigest()


def _require_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"scryfall {name} must be a non-empty string")
    return value.strip()


def _require_uuid(value: Any, name: str) -> str:
    text = _require_string(value, name)
    try:
        parsed = UUID(text)
    except ValueError as error:
        raise ValidationError(f"scryfall {name} must be a UUID") from error
    canonical = str(parsed)
    if text.lower() != canonical:
        raise ValidationError(f"scryfall {name} must be a canonical UUID")
    return canonical


def _require_positive_decimal(value: Any, name: str) -> str:
    text = str(value)
    if not text.isascii() or not text.isdecimal() or int(text) <= 0:
        raise ValidationError(f"scryfall {name} must be a positive decimal integer")
    return str(int(text))
