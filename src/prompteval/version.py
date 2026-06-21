"""Single source of truth for the package version.

Kept separate from `__init__.py` so importing the version doesn't trigger
the rest of the package's imports — useful for build tooling and
shells that want a cheap version check.
"""

__version__ = "0.1.0"
