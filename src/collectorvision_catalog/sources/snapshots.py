from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from ..artifacts import SourceRevision

T = TypeVar("T")


@dataclass(frozen=True)
class SourceSnapshot(Generic[T]):
    revision: SourceRevision
    rows: tuple[T, ...]
