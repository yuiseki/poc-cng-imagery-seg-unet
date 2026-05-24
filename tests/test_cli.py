"""CLI surface: recipe inspect + recipe-loadable file paths."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from imagery_seg.cli import app

runner = CliRunner()

RECIPES = Path(__file__).resolve().parent.parent / "recipes"


def test_cli_help_lists_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    out = result.stdout
    assert "inspect" in out
    assert "train" in out
    assert "eval" in out
    assert "sweep" in out


def test_cli_inspect_hotosm_recipe():
    result = runner.invoke(app, ["inspect", str(RECIPES / "hotosm_buildings_yugawara.py")])
    assert result.exit_code == 0, result.stdout
    out = result.stdout
    assert "hotosm" in out
    assert "overpass" in out
    assert "building" in out
    assert "train AOIs" in out
    assert "val AOIs" in out


def test_cli_inspect_sentinel2_recipe():
    result = runner.invoke(app, ["inspect", str(RECIPES / "sentinel2_parks.py")])
    assert result.exit_code == 0, result.stdout
    out = result.stdout
    assert "sentinel2" in out
    assert "park" in out


def test_cli_inspect_hotosm_park_recipe():
    result = runner.invoke(app, ["inspect", str(RECIPES / "hotosm_parks_sagamihara.py")])
    assert result.exit_code == 0, result.stdout
    out = result.stdout
    assert "hotosm" in out
    assert "overpass" in out
    assert "park" in out
    assert "train AOIs" in out
    assert "val AOIs" in out


def test_cli_inspect_hotosm_park_inagi_recipe():
    result = runner.invoke(app, ["inspect", str(RECIPES / "hotosm_parks_inagi.py")])
    assert result.exit_code == 0, result.stdout
    out = result.stdout
    assert "hotosm" in out
    assert "overpass" in out
    assert "park" in out
    assert "train AOIs" in out
    assert "val AOIs" in out


def test_cli_inspect_hotosm_parking_inagi_recipe():
    result = runner.invoke(app, ["inspect", str(RECIPES / "hotosm_parking_inagi.py")])
    assert result.exit_code == 0, result.stdout
    out = result.stdout
    assert "hotosm" in out
    assert "overpass" in out
    assert "parking" in out
    assert "train AOIs" in out
    assert "val AOIs" in out
