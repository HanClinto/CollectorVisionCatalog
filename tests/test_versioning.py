from __future__ import annotations

import pytest

from collectorvision_catalog import (
    ValidationError,
    plan_catalog_version,
    version_root,
)


def test_catalog_versions_begin_at_zero_and_checkpoint_every_ten() -> None:
    initial = plan_catalog_version(None)
    assert (initial.version, initial.publish_base, initial.publish_delta) == (0, True, False)

    incremental = plan_catalog_version(8)
    assert (incremental.version, incremental.publish_base, incremental.publish_delta) == (
        9,
        False,
        True,
    )

    checkpoint = plan_catalog_version(9)
    assert (checkpoint.version, checkpoint.publish_base, checkpoint.publish_delta) == (
        10,
        True,
        True,
    )


def test_hard_checkpoint_omits_the_delta() -> None:
    plan = plan_catalog_version(16, force_full_refresh=True)

    assert plan.version == 17
    assert plan.previous_version == 16
    assert plan.publish_base
    assert not plan.publish_delta


def test_checkpoint_interval_is_configurable() -> None:
    assert plan_catalog_version(4, checkpoint_interval=5).publish_base
    assert not plan_catalog_version(3, checkpoint_interval=5).publish_base


def test_public_paths_are_catalog_local_and_self_describing() -> None:
    assert version_root("scryfall-mtg", 10) == "scryfall-mtg/version/10"


@pytest.mark.parametrize(
    ("function", "args"),
    [
        (version_root, ("Scryfall MTG", 1)),
        (version_root, ("scryfall-mtg", -1)),
    ],
)
def test_public_paths_reject_ambiguous_inputs(function, args) -> None:
    with pytest.raises(ValidationError):
        function(*args)
