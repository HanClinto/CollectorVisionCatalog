from __future__ import annotations

from pathlib import Path

from conftest import TrackingEmbedder, TrackingImageLoader, make_row

from collectorvision_catalog import build_catalog, manifest_filename_for_catalog
from collectorvision_catalog.cli import main


def test_cli_inspect_and_validate(workspace: Path, capsys) -> None:
    build_catalog(
        [make_row("alpha", "memory://alpha", "fp-alpha", metadata={"name": "Alpha"})],
        embedder=TrackingEmbedder(),
        image_loader=TrackingImageLoader({"memory://alpha": (255, 0, 0)}),
        output_dir=workspace / "build",
        catalog_key="demo/catalog",
        version="v1",
        embedding_model="milo1",
    )
    manifest_path = workspace / "build" / manifest_filename_for_catalog("demo/catalog")

    assert main(["inspect", str(manifest_path)]) == 0
    inspect_output = capsys.readouterr().out
    assert '"schema_version": 2' in inspect_output
    assert '"catalog_key": "demo/catalog"' in inspect_output
    assert '"embedding_model": "milo1"' in inspect_output

    assert main(["validate", str(manifest_path), "--asset-dir", str(workspace / "build")]) == 0
    validate_output = capsys.readouterr().out
    assert "Validated demo/catalog@v1 (1 rows, dim=4)" in validate_output
