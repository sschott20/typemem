import time
import pytest
from typemem.plugins.base import ObservationPlugin
from typemem.plugins.runner import ObservationRunner


class CounterPlugin(ObservationPlugin):
    def __init__(self):
        self.run_count = 0
        self.setup_called = False
        self.teardown_called = False

    @property
    def name(self):
        return "counter"

    @property
    def interval_seconds(self):
        return 0.1

    def setup(self, memory_manager, robot_id):
        self.setup_called = True
        self._manager = memory_manager
        self._robot_id = robot_id

    def run(self):
        self.run_count += 1
        return []

    def teardown(self):
        self.teardown_called = True


class TestObservationRunner:
    def test_register(self):
        runner = ObservationRunner()
        plugin = CounterPlugin()
        runner.register(plugin)
        assert "counter" in runner.list_plugins()

    def test_start_calls_setup(self, manager):
        runner = ObservationRunner()
        plugin = CounterPlugin()
        runner.register(plugin)
        runner.start(manager, "robot1", tick_interval=0.05)
        time.sleep(0.1)
        runner.stop()
        assert plugin.setup_called

    def test_stop_calls_teardown(self, manager):
        runner = ObservationRunner()
        plugin = CounterPlugin()
        runner.register(plugin)
        runner.start(manager, "robot1", tick_interval=0.05)
        time.sleep(0.1)
        runner.stop()
        assert plugin.teardown_called

    def test_plugin_runs_periodically(self, manager):
        runner = ObservationRunner()
        plugin = CounterPlugin()
        runner.register(plugin)
        runner.start(manager, "robot1", tick_interval=0.05)
        time.sleep(0.5)
        runner.stop()
        assert plugin.run_count >= 2

    def test_error_in_plugin_doesnt_crash(self, manager):
        class FailPlugin(ObservationPlugin):
            @property
            def name(self):
                return "fail"
            @property
            def interval_seconds(self):
                return 0.1
            def run(self):
                raise RuntimeError("oops")

        runner = ObservationRunner()
        runner.register(FailPlugin())
        runner.start(manager, "robot1", tick_interval=0.05)
        time.sleep(0.3)
        runner.stop()
