import pytest
from typemem.memory_manager import MemoryManager


@pytest.fixture
def manager(tmp_path):
    """Fresh MemoryManager per test."""
    return MemoryManager(persist_dir=str(tmp_path / "chroma"), robot_id="test_robot")
