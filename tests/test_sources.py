from __future__ import annotations

import pytest

from collectorvision_catalog import SourceRevision, ValidationError, normalize_rfc3339_utc
from collectorvision_catalog.sources import (
    build_tcgplayer_image_urls,
    is_probable_card_product,
    normalize_scryfall_card,
    normalize_tcgcsv_product,
)

CARD_ID = "00000000-0000-0000-0000-000000000123"
ORACLE_ID = "11111111-1111-1111-1111-111111111999"


def test_source_timestamp_normalization_and_strict_revision() -> None:
    assert normalize_rfc3339_utc("2026-07-24T20:11:00+0000") == "2026-07-24T20:11:00Z"
    assert normalize_rfc3339_utc("2026-07-24T22:11:00+02:00") == (
        "2026-07-24T20:11:00Z"
    )
    with pytest.raises(ValidationError, match="normalized"):
        SourceRevision(
            source_type="tcgcsv",
            source_name="tcgplayer",
            updated_at="2026-07-24T20:11:00+00:00",
            uri="https://tcgcsv.com/last-updated.txt",
            identity="revision",
        )


def test_normalize_scryfall_card_faces_and_identifiers() -> None:
    card = {
        "id": CARD_ID,
        "name": "Sample Card",
        "oracle_id": ORACLE_ID,
        "tcgplayer_id": 1001,
        "tcgplayer_etched_id": 2002,
        "set": "neo",
        "set_name": "Neo Genesis",
        "lang": "en",
        "collector_number": "15",
        "rarity": "rare",
        "finishes": ["nonfoil", "foil"],
        "cmc": 4.0,
        "colors": ["W", "U"],
        "layout": "transform",
        "promo": True,
        "card_faces": [
            {
                "name": "Front Face",
                "cmc": 2.0,
                "colors": ["W"],
                "image_uris": {
                    "normal": "https://img/front-normal.png",
                    "png": "https://img/front.png",
                },
            },
            {
                "name": "Back Face",
                "image_uris": {
                    "large": "https://img/back-large.png",
                },
            },
        ],
    }

    rows = normalize_scryfall_card(card)

    assert [row.key for row in rows] == [
        f"scryfall:{CARD_ID}",
        f"scryfall:{CARD_ID}:face:1",
    ]
    assert rows[0].face_index == 0
    assert rows[1].face_index == 1
    assert "face_index" not in rows[0].minimal_record()
    assert rows[1].minimal_record()["face_index"] == 1
    assert rows[0].image_url == "https://img/front.png"
    assert rows[1].image_url == "https://img/back-large.png"
    assert rows[0].identifiers == {
        "scryfall_oracle": ORACLE_ID,
        "tcgplayer_etched_product": "2002",
        "tcgplayer_product": "1001",
    }
    assert rows[0].finishes == ("foil", "nonfoil")
    assert rows[0].minimal_record()["finishes"] == ["foil", "nonfoil"]
    assert "finishes" not in rows[0].metadata
    assert rows[0].metadata["cmc"] == 2
    assert rows[0].metadata["colors"] == ["W"]
    assert rows[0].metadata["layout"] == "transform"
    assert rows[0].metadata["promo"] is True
    assert rows[1].metadata == {
        "cmc": 4,
        "collector_number": "15",
        "colors": ["W", "U"],
        "lang": "en",
        "layout": "transform",
        "name": "Back Face",
        "promo": True,
        "rarity": "rare",
        "set": "neo",
        "set_name": "Neo Genesis",
    }


def test_scryfall_floors_fractional_joke_card_cmc() -> None:
    row = normalize_scryfall_card(
        {
            "id": CARD_ID,
            "name": "Little Joke Card",
            "cmc": 0.5,
            "layout": "normal",
            "promo": False,
            "image_uris": {"png": "https://img/card.png"},
        }
    )[0]

    assert row.metadata["cmc"] == 0
    assert isinstance(row.metadata["cmc"], int)


def test_scryfall_image_revision_changes_fingerprint() -> None:
    card = {
        "id": CARD_ID,
        "name": "Sample Card",
        "layout": "normal",
        "promo": False,
        "image_uris": {"png": "https://cards.scryfall.io/png/front/a/b/card.png?100"},
    }
    first = normalize_scryfall_card(card)[0]
    card["image_uris"]["png"] = "https://cards.scryfall.io/png/front/a/b/card.png?200"
    second = normalize_scryfall_card(card)[0]
    assert first.image_fingerprint != second.image_fingerprint


@pytest.mark.parametrize(
    "finishes",
    [
        "foil",
        ["foil", "foil"],
        [" "],
    ],
)
def test_scryfall_rejects_invalid_finishes(finishes: object) -> None:
    with pytest.raises(ValidationError, match="finishe?s?"):
        normalize_scryfall_card(
            {
                "id": CARD_ID,
                "name": "Sample Card",
                "finishes": finishes,
                "layout": "normal",
                "promo": False,
                "image_uris": {"png": "https://img/card.png"},
            }
        )


def test_normalize_tcgcsv_product_images_and_filtering() -> None:
    category = {"categoryId": 10, "name": "Pokemon Singles"}
    group = {"groupId": 20, "name": "Base Set"}
    product = {
        "productId": 123456,
        "name": "Charizard",
        "imageCount": 2,
        "extendedData": [
            {"name": "Number", "value": "4/102"},
            {"name": "Rarity", "value": "Rare Holo"},
            {"name": "Language", "value": "English"},
            {"name": "Printing", "value": "Reverse Holo"},
        ],
    }

    assert is_probable_card_product(product) is True
    rows = normalize_tcgcsv_product(product, group=group, category=category)

    assert build_tcgplayer_image_urls("123456", 2) == [
        "https://tcgplayer-cdn.tcgplayer.com/product/123456_in_1000x1000.jpg",
        "https://tcgplayer-cdn.tcgplayer.com/product/123456_1_in_1000x1000.jpg",
    ]
    assert [row.image_url for row in rows] == build_tcgplayer_image_urls("123456", 2)
    assert rows[1].face_index == 1
    assert rows[0].identifiers == {
        "tcgplayer_category": "10",
        "tcgplayer_group": "20",
    }
    assert rows[0].metadata == {
        "category": "Pokemon Singles",
        "collector_number": "4/102",
        "language": "English",
        "name": "Charizard",
        "rarity": "Rare Holo",
        "set": "Base Set",
    }

    not_a_card = {
        "productId": 999,
        "name": "Deck Box",
        "imageCount": 1,
        "extendedData": [{"name": "Color", "value": "Blue"}],
    }
    assert is_probable_card_product(not_a_card) is False
    assert normalize_tcgcsv_product(not_a_card, group=group, category=category) == []


def test_tcgcsv_does_not_infer_available_finishes_from_product_data() -> None:
    rows = normalize_tcgcsv_product(
        {
            "productId": 123456,
            "name": "Charizard",
            "imageCount": 1,
            "extendedData": [
                {"name": "Number", "value": "4/102"},
                {"name": "Printing", "value": "Reverse Holo"},
                {"name": "Finish", "value": "Foil"},
            ],
        }
    )

    assert "finishes" not in rows[0].metadata
    assert "foilings" not in rows[0].metadata


def test_tcgcsv_product_revision_changes_image_fingerprint() -> None:
    product = {
        "productId": 123456,
        "name": "Charizard",
        "modifiedOn": "2026-05-01T00:00:00",
        "extendedData": [{"name": "Number", "value": "4/102"}],
    }
    first = normalize_tcgcsv_product(product)[0]
    product["modifiedOn"] = "2026-05-02T00:00:00"
    second = normalize_tcgcsv_product(product)[0]
    assert first.image_fingerprint != second.image_fingerprint
    assert "modified_on" not in second.metadata


def test_tcgcsv_product_without_images_is_skipped() -> None:
    product = {
        "productId": 123456,
        "name": "Unavailable Card",
        "imageCount": 0,
        "extendedData": [{"name": "Number", "value": "1"}],
    }
    assert normalize_tcgcsv_product(product) == []


def test_source_identifiers_reject_path_components() -> None:
    with pytest.raises(ValidationError, match="must be a UUID"):
        normalize_scryfall_card(
            {
                "id": "../../escape",
                "name": "Unsafe",
                "image_uris": {"png": "https://example.test/card.png"},
            }
        )
    with pytest.raises(ValidationError, match="positive decimal"):
        normalize_tcgcsv_product(
            {
                "productId": "../../escape",
                "name": "Unsafe",
                "imageCount": 1,
                "extendedData": [{"name": "Number", "value": "1"}],
            }
        )
