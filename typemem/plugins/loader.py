import importlib
import inspect
import logging
import pkgutil
from pathlib import Path
from typing import List, Optional, Set

from typemem.plugins.base import ObservationPlugin, ConsolidationPlugin

logger = logging.getLogger(__name__)


class PluginLoader:
    """Auto-discovers and loads observation and consolidation plugins."""

    @staticmethod
    def discover(
        obs_package: Optional[str] = None,
        consol_package: Optional[str] = None,
        disabled: Optional[List[str]] = None,
    ) -> tuple[List[ObservationPlugin], List[ConsolidationPlugin]]:
        disabled_set: Set[str] = set(disabled or [])

        obs_plugins = []
        if obs_package:
            obs_plugins = PluginLoader._scan_package(
                obs_package, ObservationPlugin, disabled_set,
            )

        consol_plugins = []
        if consol_package:
            consol_plugins = PluginLoader._scan_package(
                consol_package, ConsolidationPlugin, disabled_set,
            )

        return obs_plugins, consol_plugins

    @staticmethod
    def _scan_package(package_name, base_class, disabled):
        plugins = []
        try:
            package = importlib.import_module(package_name)
        except ImportError:
            logger.warning("Plugin package %s not found", package_name)
            return plugins

        package_path = Path(package.__file__).parent

        for module_info in pkgutil.iter_modules([str(package_path)]):
            module_name = f"{package_name}.{module_info.name}"
            try:
                module = importlib.import_module(module_name)
            except Exception as e:
                logger.error("Failed to import plugin module %s: %s", module_name, e)
                continue

            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (
                    issubclass(obj, base_class)
                    and obj is not base_class
                    and not inspect.isabstract(obj)
                ):
                    sig = inspect.signature(obj.__init__)
                    required = [
                        p for p in list(sig.parameters.values())[1:]
                        if p.default is inspect.Parameter.empty
                        and p.kind not in (
                            inspect.Parameter.VAR_POSITIONAL,
                            inspect.Parameter.VAR_KEYWORD,
                        )
                    ]
                    if required:
                        continue

                    try:
                        instance = obj()
                        if instance.name in disabled:
                            logger.info("Plugin '%s' disabled, skipping", instance.name)
                            continue
                        plugins.append(instance)
                        logger.info("Loaded plugin '%s' from %s", instance.name, module_name)
                    except Exception as e:
                        logger.error("Failed to instantiate %s from %s: %s", name, module_name, e)

        return plugins
