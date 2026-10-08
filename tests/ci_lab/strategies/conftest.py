"""Reuse the optimizer test fakes (KeywordDomain, git worktree fixture)."""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("_ci_optim_fakes", Path(__file__).parents[1] / "optim" / "conftest.py")
_fakes = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fakes)

KeywordDomain = _fakes.KeywordDomain
git = _fakes.git
worktree = _fakes.worktree
domain = _fakes.domain
