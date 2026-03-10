import pytest
from typemem.plugins.loader import PluginLoader


class TestPluginLoader:
    def test_discover_with_no_packages(self):
        obs, consol = PluginLoader.discover()
        assert obs == []
        assert consol == []

    def test_discover_nonexistent_package(self):
        obs, consol = PluginLoader.discover(obs_package="nonexistent.package")
        assert obs == []

    def test_discover_with_disabled(self):
        obs, consol = PluginLoader.discover(disabled=["some_plugin"])
        assert obs == []
