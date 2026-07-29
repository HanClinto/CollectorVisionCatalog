from __future__ import annotations

import re
from dataclasses import dataclass

from .artifacts import ValidationError

_PUBLIC_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
@dataclass(frozen=True)
class CatalogVersionPlan:
    version: int
    previous_version: int | None
    publish_base: bool
    publish_delta: bool


def plan_catalog_version(
    previous_version: int | None,
    *,
    checkpoint_interval: int = 10,
    force_full_refresh: bool = False,
) -> CatalogVersionPlan:
    """Plan the next changed version for one catalog."""
    if checkpoint_interval <= 0:
        raise ValidationError("checkpoint_interval must be positive")
    if previous_version is not None and (
        isinstance(previous_version, bool) or previous_version < 0
    ):
        raise ValidationError("previous catalog version must be a non-negative integer or null")
    if previous_version is None:
        return CatalogVersionPlan(
            version=0,
            previous_version=None,
            publish_base=True,
            publish_delta=False,
        )

    version = previous_version + 1
    hard_checkpoint = bool(force_full_refresh)
    return CatalogVersionPlan(
        version=version,
        previous_version=previous_version,
        publish_base=hard_checkpoint or version % checkpoint_interval == 0,
        publish_delta=not hard_checkpoint,
    )


def validate_public_name(value: str) -> str:
    if not isinstance(value, str) or _PUBLIC_NAME.fullmatch(value) is None:
        raise ValidationError("catalog public_name must be lowercase kebab-case")
    return value


def version_root(public_name: str, version: int) -> str:
    validate_public_name(public_name)
    _validate_version(version)
    return f"{public_name}/version/{version}"


def _validate_version(version: int) -> None:
    if not isinstance(version, int) or isinstance(version, bool) or version < 0:
        raise ValidationError("catalog version must be a non-negative integer")
