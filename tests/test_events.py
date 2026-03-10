import collections
import threading
import pytest
from typemem import events


@pytest.fixture(autouse=True)
def reset_event_bus():
    """Reset event bus state between tests."""
    events._registry.clear()
    yield
    events._registry.clear()


class TestEventBus:
    def test_subscribe_returns_deque(self):
        dq = events.subscribe("test_channel")
        assert isinstance(dq, collections.deque)

    def test_push_to_subscriber(self):
        dq = events.subscribe("sensor")
        events.push("sensor", {"value": 42})
        assert len(dq) == 1
        assert dq[0] == {"value": 42}

    def test_push_without_subscriber_drops(self):
        events.push("no_sub", {"data": 1})

    def test_fan_out_multiple_subscribers(self):
        dq1 = events.subscribe("events")
        dq2 = events.subscribe("events")
        events.push("events", {"x": 1})
        assert len(dq1) == 1
        assert len(dq2) == 1
        assert dq1[0] == dq2[0]

    def test_channel_isolation(self):
        dq_a = events.subscribe("chan_a")
        dq_b = events.subscribe("chan_b")
        events.push("chan_a", {"a": 1})
        assert len(dq_a) == 1
        assert len(dq_b) == 0

    def test_maxlen_respected(self):
        dq = events.subscribe("bounded", maxlen=3)
        for i in range(5):
            events.push("bounded", {"i": i})
        assert len(dq) == 3
        assert dq[0] == {"i": 2}

    def test_channels_list(self):
        events.subscribe("alpha")
        events.subscribe("beta")
        ch = events.channels()
        assert "alpha" in ch
        assert "beta" in ch

    def test_thread_safety(self):
        dq = events.subscribe("threaded")
        errors = []
        def pusher():
            try:
                for i in range(100):
                    events.push("threaded", {"i": i})
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=pusher) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert len(dq) == 400

    def test_drain_pattern(self):
        dq = events.subscribe("drain")
        events.push("drain", {"a": 1})
        events.push("drain", {"b": 2})
        drained = []
        while dq:
            drained.append(dq.popleft())
        assert len(drained) == 2
        assert len(dq) == 0
