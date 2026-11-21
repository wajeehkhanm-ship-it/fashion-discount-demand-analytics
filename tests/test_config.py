"""Tests that config/settings.yaml loads and holds sane values."""

from __future__ import annotations

from fdda.config import data_mode, load_settings, resolve_path


def test_settings_load_and_paths_resolve():
    settings = load_settings()
    assert (settings["_root"] / "config" / "settings.yaml").exists()
    assert resolve_path(settings, "raw").name == "raw"
    assert data_mode(settings) in {"dev", "full"}


def test_discount_bands_are_consistent():
    bands = load_settings()["pricing"]["discount_bands"]
    edges = bands["edges"]
    assert edges == sorted(edges) and edges[0] == 0
    assert len(bands["labels"]) == len(edges) - 1


def test_seed_and_sample_fraction_set():
    settings = load_settings()
    assert isinstance(settings["project"]["seed"], int)
    assert 0 < settings["dev_sample_fraction"] <= 1
