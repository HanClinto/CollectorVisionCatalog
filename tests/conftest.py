from __future__ import annotations

import io
import re
import shutil
import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from collectorvision_catalog import PrimaryID, RecognitionRow  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _clean_session_workspace() -> Iterable[None]:
    workspace_root = ROOT / ".pytest-workspace"
    if workspace_root.exists():
        shutil.rmtree(workspace_root)
    yield
    if workspace_root.exists():
        shutil.rmtree(workspace_root)


@pytest.fixture
def workspace(request: pytest.FixtureRequest) -> Iterable[Path]:
    workspace_root = ROOT / ".pytest-workspace"
    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", request.node.nodeid)
    path = workspace_root / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    yield path
    if path.exists():
        shutil.rmtree(path)


class TrackingImageLoader:
    def __init__(self, images: dict[str, tuple[int, int, int]]) -> None:
        self._images = images
        self.calls: list[str] = []

    def __call__(self, image_url: str) -> Image.Image:
        self.calls.append(image_url)
        color = self._images[image_url]
        return Image.open(io.BytesIO(make_png_bytes(color))).convert("RGB")


class TrackingEmbedder:
    def __init__(self) -> None:
        self.calls: list[list[tuple[int, int, int]]] = []

    def __call__(self, images: list[Image.Image]) -> np.ndarray:
        colors: list[tuple[int, int, int]] = []
        vectors: list[np.ndarray] = []
        for image in images:
            pixel = tuple(int(value) for value in image.getpixel((0, 0)))
            colors.append(pixel)
            vector = np.array(
                [pixel[0] / 255.0, pixel[1] / 255.0, pixel[2] / 255.0, 0.5],
                dtype=np.float32,
            )
            vector /= np.linalg.norm(vector)
            vectors.append(vector)
        self.calls.append(colors)
        return np.vstack(vectors)


class BadEmbedder:
    def __call__(self, images: list[Image.Image]) -> np.ndarray:
        return np.full((len(images), 3), np.nan, dtype=np.float32)


class UnnormalizedEmbedder:
    def __call__(self, images: list[Image.Image]) -> np.ndarray:
        return np.ones((len(images), 3), dtype=np.float32)


def make_png_bytes(color: tuple[int, int, int]) -> bytes:
    image = Image.new("RGB", (2, 2), color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def make_row(
    key: str,
    image_url: str,
    image_fingerprint: str,
    *,
    namespace: str = "test",
    primary_value: str | None = None,
    secondary_ids: dict[str, str] | None = None,
    face_index: int = 0,
    metadata: dict[str, object] | None = None,
) -> RecognitionRow:
    return RecognitionRow(
        key=key,
        primary_id=PrimaryID(namespace=namespace, value=primary_value or key),
        secondary_ids=secondary_ids or {},
        face_index=face_index,
        image_url=image_url,
        image_fingerprint=image_fingerprint,
        metadata=metadata,
    )
