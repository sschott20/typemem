import pytest
from typemem.chromadb_store import ChromaDBStore


@pytest.fixture
def store(tmp_path):
    """Fresh ChromaDB store per test."""
    return ChromaDBStore(persist_dir=str(tmp_path / "chroma"))
