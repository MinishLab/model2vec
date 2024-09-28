from pathlib import Path
from tempfile import NamedTemporaryFile

from model2vec.utils import _get_metadata_from_readme


def test__get_metadata_from_readme_not_exists() -> None:
    """Test getting metadata from a README."""
    assert _get_metadata_from_readme(Path("zzz")) == {}


def test__get_metadata_from_readme_mocked_file() -> None:
    """Test getting metadata from a README."""
    with NamedTemporaryFile() as f:
        f.write(b"---\nkey: value\n---\n")
        f.flush()
        assert _get_metadata_from_readme(Path(f.name))["key"] == "value"


def test__get_metadata_from_readme_mocked_file_keys() -> None:
    """Test getting metadata from a README."""
    with NamedTemporaryFile() as f:
        f.write(b"")
        f.flush()
        assert set(_get_metadata_from_readme(Path(f.name))) == set()
