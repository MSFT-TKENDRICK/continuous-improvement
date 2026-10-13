"""MCP integration for the harness: frozen server registry, evolvable exposure, hub, direct and code mode.

Code mode is an isolation layer, not an OS sandbox (see :mod:`ci_lab.mcp.codemode` and docs/mcp.md).
"""

from __future__ import annotations

from typing import Any

from ci_lab.mcp.registry import (
    EXPOSURE_FORMAT,
    FROZEN_SERVERS,
    REGISTRY_FORMAT,
    CodeModeCaps,
    CodeModeConfig,
    Exposure,
    McpConfigError,
    Registry,
    ServerDef,
    ServerExposure,
    exposure_path,
    load_exposure,
    load_registry,
    validate_exposure,
)

_LAZY = {
    "McpHub": "ci_lab.mcp.client", "McpDenied": "ci_lab.mcp.client", "McpToolError": "ci_lab.mcp.client",
    "ToolInfo": "ci_lab.mcp.client", "maf_tools": "ci_lab.mcp.client",
    "CodeMode": "ci_lab.mcp.codemode", "check_code": "ci_lab.mcp.codemode",
}

__all__ = [
    "EXPOSURE_FORMAT",
    "FROZEN_SERVERS",
    "REGISTRY_FORMAT",
    "CodeMode",
    "CodeModeCaps",
    "CodeModeConfig",
    "Exposure",
    "McpConfigError",
    "McpDenied",
    "McpHub",
    "McpToolError",
    "Registry",
    "ServerDef",
    "ServerExposure",
    "ToolInfo",
    "check_code",
    "exposure_path",
    "load_exposure",
    "load_registry",
    "maf_tools",
    "validate_exposure",
]


def __getattr__(name: str) -> Any:  # lazy: the MCP SDK is only imported when the hub is used
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(name)
