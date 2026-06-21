"""First test — proves the package builds, imports, and exposes a version.

Keep this test green at all times; if it ever fails the project skeleton
itself is broken and nothing downstream matters yet.
"""

from prompteval import __version__


def test_version_string_is_pep440() -> None:
    # PEP 440 versions are at minimum 'X.Y.Z'. Tight check on purpose so a
    # future copy-paste of an invalid version string fails fast.
    parts = __version__.split(".")
    assert len(parts) >= 3
    for part in parts[:3]:
        assert part.isdigit(), f"Version part is not numeric: {part!r}"
