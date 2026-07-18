from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import RecognitionRow, ValidationError

_DECISIONS = {"approve", "quarantine", "reject"}
_MATCH_FIELDS = {
    "primary_namespace",
    "primary_id",
    "category_id",
    "group_id",
    "face_index",
    "name_regex",
}


@dataclass(frozen=True)
class QualityRule:
    rule_id: str
    source_type: str
    decision: str
    match: dict[str, str | int]
    exclude_matches: tuple[dict[str, str | int], ...]
    reason: str
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class QualityFinding:
    key: str
    rule_id: str
    decision: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {
            "key": self.key,
            "rule_id": self.rule_id,
            "decision": self.decision,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class QualityResult:
    rows: tuple[RecognitionRow, ...]
    findings: tuple[QualityFinding, ...]

    def report(self) -> dict[str, Any]:
        counts = Counter(finding.rule_id for finding in self.findings)
        return {
            "excluded_rows": len(self.findings),
            "rules": [
                {"rule_id": rule_id, "excluded_rows": count}
                for rule_id, count in sorted(counts.items())
            ],
            "findings": [
                finding.to_dict()
                for finding in sorted(self.findings, key=lambda finding: finding.key)
            ],
        }


def load_quality_rules(path: Path) -> tuple[QualityRule, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValidationError("quality override schema_version must be 1")
    raw_rules = payload.get("rules")
    if not isinstance(raw_rules, list):
        raise ValidationError("quality override rules must be a list")

    rules: list[QualityRule] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(raw_rules):
        if not isinstance(raw, Mapping):
            raise ValidationError(f"quality rule {index} must be an object")
        rule_id = _required_text(raw.get("id"), f"quality rule {index}.id")
        if rule_id in seen_ids:
            raise ValidationError(f"duplicate quality rule ID {rule_id!r}")
        seen_ids.add(rule_id)
        decision = _required_text(raw.get("decision"), f"quality rule {rule_id}.decision")
        if decision not in _DECISIONS:
            raise ValidationError(
                f"quality rule {rule_id!r} decision must be one of {sorted(_DECISIONS)}"
            )
        match = _parse_match(raw.get("match"), rule_id, "match")
        raw_exclude_matches = raw.get("exclude_matches", [])
        if not isinstance(raw_exclude_matches, list):
            raise ValidationError(
                f"quality rule {rule_id!r} exclude_matches must be a list"
            )
        exclude_matches = tuple(
            _parse_match(raw_exclude, rule_id, f"exclude_matches[{exclude_index}]")
            for exclude_index, raw_exclude in enumerate(raw_exclude_matches)
        )
        raw_evidence = raw.get("evidence", [])
        if not isinstance(raw_evidence, list):
            raise ValidationError(f"quality rule {rule_id!r} evidence must be a list")
        rules.append(
            QualityRule(
                rule_id=rule_id,
                source_type=_required_text(
                    raw.get("source_type"),
                    f"quality rule {rule_id}.source_type",
                ),
                decision=decision,
                match=match,
                exclude_matches=exclude_matches,
                reason=_required_text(raw.get("reason"), f"quality rule {rule_id}.reason"),
                evidence=tuple(
                    _required_text(item, f"quality rule {rule_id}.evidence")
                    for item in raw_evidence
                ),
            )
        )
    return tuple(rules)


def apply_quality_rules(
    rows: Iterable[RecognitionRow],
    *,
    source_type: str,
    rules: Iterable[QualityRule],
) -> QualityResult:
    source_rules = tuple(rule for rule in rules if rule.source_type == source_type)
    kept: list[RecognitionRow] = []
    findings: list[QualityFinding] = []
    for row in rows:
        matching = [rule for rule in source_rules if _matches(rule, row)]
        decisions = {rule.decision for rule in matching}
        if len(decisions) > 1:
            raise ValidationError(
                f"quality rules conflict for {row.key!r}: "
                f"{sorted(rule.rule_id for rule in matching)}"
            )
        if matching and matching[0].decision in {"quarantine", "reject"}:
            rule = matching[0]
            findings.append(
                QualityFinding(
                    key=row.key,
                    rule_id=rule.rule_id,
                    decision=rule.decision,
                    reason=rule.reason,
                )
            )
        else:
            kept.append(row)
    return QualityResult(rows=tuple(kept), findings=tuple(findings))


def _matches(rule: QualityRule, row: RecognitionRow) -> bool:
    return _matches_fields(rule.match, row) and not any(
        _matches_fields(exclude_match, row)
        for exclude_match in rule.exclude_matches
    )


def _matches_fields(match: Mapping[str, str | int], row: RecognitionRow) -> bool:
    values: dict[str, str | int | None] = {
        "primary_namespace": row.primary_id.namespace,
        "primary_id": row.primary_id.value,
        "category_id": row.secondary_ids.get("tcgplayer_category"),
        "group_id": row.secondary_ids.get("tcgplayer_group"),
        "face_index": row.face_index,
    }
    for field, expected in match.items():
        if field == "name_regex":
            name = row.metadata.get("name")
            if not isinstance(name, str) or re.search(str(expected), name) is None:
                return False
        elif values[field] != expected:
            return False
    return True


def _parse_match(
    raw_match: Any,
    rule_id: str,
    field_name: str,
) -> dict[str, str | int]:
    if not isinstance(raw_match, Mapping) or not raw_match:
        raise ValidationError(
            f"quality rule {rule_id!r} {field_name} must be a non-empty object"
        )
    unknown_fields = set(raw_match).difference(_MATCH_FIELDS)
    if unknown_fields:
        raise ValidationError(
            f"quality rule {rule_id!r} {field_name} has unknown fields: "
            f"{sorted(unknown_fields)}"
        )
    return {
        field: _normalize_match_value(field, value, rule_id)
        for field, value in raw_match.items()
    }


def _normalize_match_value(field: str, value: Any, rule_id: str) -> str | int:
    if field == "face_index":
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValidationError(
                f"quality rule {rule_id!r} face_index must be a non-negative integer"
            )
        return value
    normalized = _required_text(str(value), f"quality rule {rule_id}.{field}")
    if field == "name_regex":
        try:
            re.compile(normalized)
        except re.error as error:
            raise ValidationError(
                f"quality rule {rule_id!r} name_regex is invalid: {error}"
            ) from error
    return normalized


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{name} must be a non-empty string")
    return value.strip()
