from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import make_row

from collectorvision_catalog import ValidationError
from collectorvision_catalog.quality import apply_quality_rules, load_quality_rules


def write_rules(path: Path, rules: list[dict[str, object]]) -> None:
    path.write_text(
        json.dumps({"schema_version": 1, "rules": rules}),
        encoding="utf-8",
    )


def test_group_quarantine_excludes_matching_rows(tmp_path: Path) -> None:
    path = tmp_path / "quality.json"
    write_rules(
        path,
        [
            {
                "id": "annotated-group",
                "source_type": "tcgcsv",
                "decision": "quarantine",
                "match": {"category_id": 1, "group_id": 1527},
                "reason": "annotated",
                "evidence": ["https://example.test/evidence"],
            }
        ],
    )
    rows = [
        make_row(
            "tcgplayer:1:face:0",
            "memory://1",
            "fp-1",
            namespace="tcgplayer",
            identifiers={"tcgplayer_category": "1", "tcgplayer_group": "1527"},
        ),
        make_row(
            "tcgplayer:2:face:0",
            "memory://2",
            "fp-2",
            namespace="tcgplayer",
            identifiers={"tcgplayer_category": "1", "tcgplayer_group": "2"},
        ),
    ]

    result = apply_quality_rules(
        rows,
        source_type="tcgcsv",
        rules=load_quality_rules(path),
    )

    assert [row.key for row in result.rows] == ["tcgplayer:2:face:0"]
    assert result.report() == {
        "excluded_rows": 1,
        "rules": [{"rule_id": "annotated-group", "excluded_rows": 1}],
        "findings": [
            {
                "key": "tcgplayer:1:face:0",
                "rule_id": "annotated-group",
                "decision": "quarantine",
                "reason": "annotated",
            }
        ],
    }


def test_group_quarantine_can_preserve_named_exceptions(tmp_path: Path) -> None:
    path = tmp_path / "quality.json"
    write_rules(
        path,
        [
            {
                "id": "annotated-group",
                "source_type": "tcgcsv",
                "decision": "quarantine",
                "match": {"category_id": 1, "group_id": 2198},
                "exclude_matches": [
                    {"name_regex": "(?i)(biography|decklist|blank) card$"}
                ],
                "reason": "annotated",
            }
        ],
    )
    rows = [
        make_row(
            "tcgplayer:1:face:0",
            "memory://1",
            "fp-1",
            namespace="tcgplayer",
            identifiers={"tcgplayer_category": "1", "tcgplayer_group": "2198"},
            metadata={"name": "1996 World Championship Blank Card"},
        ),
        make_row(
            "tcgplayer:2:face:0",
            "memory://2",
            "fp-2",
            namespace="tcgplayer",
            identifiers={"tcgplayer_category": "1", "tcgplayer_group": "2198"},
            metadata={"name": "Black Lotus - 1996"},
        ),
    ]

    result = apply_quality_rules(
        rows,
        source_type="tcgcsv",
        rules=load_quality_rules(path),
    )

    assert [row.key for row in result.rows] == ["tcgplayer:1:face:0"]
    assert [finding.key for finding in result.findings] == ["tcgplayer:2:face:0"]


def test_conflicting_quality_decisions_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "quality.json"
    common = {
        "source_type": "tcgcsv",
        "match": {"identifiers": {"tcgplayer": "1"}},
        "reason": "reviewed",
    }
    write_rules(
        path,
        [
            {"id": "approve", "decision": "approve", **common},
            {"id": "reject", "decision": "reject", **common},
        ],
    )
    row = make_row(
        "tcgplayer:1:face:0",
        "memory://1",
        "fp-1",
        namespace="tcgplayer",
        primary_value="1",
    )

    with pytest.raises(ValidationError, match="quality rules conflict"):
        apply_quality_rules(
            [row],
            source_type="tcgcsv",
            rules=load_quality_rules(path),
        )
