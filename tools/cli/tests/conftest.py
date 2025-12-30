#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# type: ignore[reportUnusedImport]

import pytest
import os
import sys
from pathlib import Path

# Add the parent directory to the path to import shared utilities
sys.path.insert(
    0, str(Path(__file__).parent.parent.parent.parent / "tools" / "server" / "tests")
)
