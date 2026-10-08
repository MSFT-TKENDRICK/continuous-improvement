"""Shared caches: CoW detection, shared uv/pycache env, venv provisioning, eval result cache."""

from ci_lab.cache.cow import CowInfo, cow_mode, detect
from ci_lab.cache.env import cache_root, shared_env, with_shared_env
from ci_lab.cache.evalcache import EvalCache, pin_hash
from ci_lab.cache.venv import Provisioned, provision

__all__ = ["CowInfo", "EvalCache", "Provisioned", "cache_root", "cow_mode", "detect", "pin_hash", "provision",
           "shared_env", "with_shared_env"]
