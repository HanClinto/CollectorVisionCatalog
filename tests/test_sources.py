from __future__ import annotations

from collectorvision_catalog.sources import (
    build_tcgplayer_image_urls,
    is_probable_card_product,
    normalize_scryfall_card,
    normalize_tcgcsv_product,
)


def test_normalize_scryfall_card_faces_and_secondary_ids() -> None:
    card = {
        "id": "card-123",
        "name": "Sample Card",
        "oracle_id": "oracle-999",
        "tcgplayer_id": 1001,
        "tcgplayer_etched_id": 2002,
        "set": "neo",
        "set_name": "Neo Genesis",
        "lang": "en",
        "collector_number": "15",
        "rarity": "rare",
        "finishes": ["nonfoil", "foil"],
        "card_faces": [
            {
                "name": "Front Face",
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

    assert [row.key for row in rows] == ["scryfall:card-123:face:0", "scryfall:card-123:face:1"]
    assert rows[0].face.is_back is False
    assert rows[1].face.is_back is True
    assert rows[0].image_url == "https://img/front.png"
    assert rows[1].image_url == "https://img/back-large.png"
    assert rows[0].secondary_ids == {
        "scryfall_oracle": "oracle-999",
        "tcgplayer_etched_product": "2002",
        "tcgplayer_product": "1001",
    }
    assert rows[1].metadata == {
        "collector_number": "15",
        "finishes": ["nonfoil", "foil"],
        "lang": "en",
        "name": "Sample Card",
        "rarity": "rare",
        "set": "neo",
        "set_name": "Neo Genesis",
    }


def test_scryfall_image_revision_changes_fingerprint() -> None:
    card = {
        "id": "card-123",
        "name": "Sample Card",
        "image_uris": {"png": "https://cards.scryfall.io/png/front/a/b/card.png?100"},
    }
    first = normalize_scryfall_card(card)[0]
    card["image_uris"]["png"] = "https://cards.scryfall.io/png/front/a/b/card.png?200"
    second = normalize_scryfall_card(card)[0]
    assert first.image_fingerprint != second.image_fingerprint


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
    assert rows[1].face.is_back is True
    assert rows[1].face.name == "Charizard image 2"
    assert rows[0].secondary_ids == {
        "tcgplayer_category": "10",
        "tcgplayer_group": "20",
    }
    assert rows[0].metadata == {
        "category": "Pokemon Singles",
        "collector_number": "4/102",
        "foilings": ["Reverse Holo"],
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
