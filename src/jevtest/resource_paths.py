"""Resolve repository resources in editable and installed environments."""

from __future__ import annotations

import sys
from pathlib import Path


def resource_path(*parts: str) -> Path:
    """Return a resource path from the source checkout or an installed wheel.

    Source checkouts keep contracts/config at the repository root. Wheels
    install those directories under the environment prefix through setuptools
    data files. The first candidate also lets editable installs follow the
    checkout's single source of truth.
    """

    source_root = Path(__file__).resolve().parents[2]
    source_path = source_root.joinpath(*parts)
    if source_path.exists():
        return source_path
    return Path(sys.prefix).joinpath(*parts)
