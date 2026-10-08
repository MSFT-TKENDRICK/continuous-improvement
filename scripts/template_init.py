#!/usr/bin/env python3
"""``ci-lab template init`` for the ``template-init`` workflow. STDLIB ONLY.

Runs in the write-token job with system Python (``python3 -I -B``): no ``uv sync``, no third-party
packages. It imports only the stdlib-only modules ``ci_lab.template.{marker,codeowners,init}`` from the
checked-out commit, the same code the ``ci-lab template init`` CLI uses. Same arguments; dry run unless
``--apply``. Exit 0 on success, 2 on bad input or refusal (e.g. in the template repository itself).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ci_lab.template.init import main

if __name__ == "__main__":
    raise SystemExit(main(["--root", str(ROOT), *sys.argv[1:]]))
