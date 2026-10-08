"""YAML loaders, trusted template catalog and last-known-good bundle (design §13.6 B3/N1)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from functools import lru_cache
from importlib import resources
from pathlib import Path
from types import MappingProxyType
from typing import Any

import re2
import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from ci_lab.rules._bundle import Bundle, Problem, RuleLoadError, build_bundle
from ci_lab.rulespec import (
    TEMPLATES_RESOURCE,
    ExtractorFile,
    ExtractorSpec,
    RuleFile,
    RuleSpec,
    TemplateSpec,
)

LKG_DIRNAME = ".lkg"
LOCK_NAME = "BUNDLE.lock"


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader with YAML 1.2 booleans (``on``/``yes``/``no`` stay strings — ``on:`` is a rule key)
    that rejects duplicate mapping keys (silent overrides hide rule edits)."""


_UniqueKeyLoader.yaml_implicit_resolvers = {
    k: [(tag, rx) for tag, rx in v if tag != "tag:yaml.org,2002:bool"]
    for k, v in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_UniqueKeyLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool", re2.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"), list("tTfF"))


def _construct_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
    seen: set[Any] = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            raise yaml.constructor.ConstructorError(None, None, f"duplicate key {key!r}", key_node.start_mark)
        seen.add(key)
    return loader.construct_mapping(node, deep=deep)


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def load_yaml(text: str) -> Any:
    """Parse rule/extractor YAML the way the engine does (YAML 1.2 booleans, duplicate keys rejected)."""
    return yaml.load(text, Loader=_UniqueKeyLoader)


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _read_yaml(path: Path, probs: list[Problem]) -> tuple[Any, bytes | None]:
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        probs.append(Problem(str(path), f"cannot read file: {exc.strerror or exc}", "Check the path exists."))
        return None, None
    try:
        data = yaml.load(raw.decode("utf-8"), Loader=_UniqueKeyLoader)
    except (yaml.YAMLError, UnicodeDecodeError) as exc:
        probs.append(Problem(str(path), f"invalid YAML: {str(exc).splitlines()[0]}", "Fix the YAML syntax."))
        return None, raw
    if data is None:
        probs.append(Problem(str(path), "file is empty", "Add `schema_version: 1` and a list of entries."))
    return data, raw


def _validation_problems(path: Path, exc: ValidationError, fix: str) -> list[Problem]:
    out = []
    for err in exc.errors():
        loc = ".".join(str(x) for x in err.get("loc", ()))
        out.append(Problem(str(path), f"{loc}: {err.get('msg')}", fix))
    return out


def _load_extractors(paths: Sequence[Path], probs: list[Problem]) -> list[ExtractorSpec]:
    out: list[ExtractorSpec] = []
    for p in paths:
        data, _ = _read_yaml(Path(p), probs)
        if data is None:
            continue
        try:
            out.extend(ExtractorFile.model_validate(data).extractors)
        except ValidationError as exc:
            probs.extend(_validation_problems(Path(p), exc, "Match the ExtractorSpec schema (unknown keys rejected)."))
    return out


def load_bundle(
    rule_paths: Sequence[Path],
    extractor_paths: Sequence[Path] = (),
    *,
    templates: Mapping[str, TemplateSpec] | None = None,
) -> Bundle:
    """Load, validate and compile rule + extractor YAML files transactionally.

    Every problem (YAML, schema, unknown template/slots, RE2, duplicate ids, conflicts,
    flags without extractors) is collected; any problem => :class:`RuleLoadError`.
    """
    probs: list[Problem] = []
    rules: list[RuleSpec] = []
    origins: dict[str, str] = {}
    sources: list[Path] = []
    digests: list[str] = []
    for p in rule_paths:
        p = Path(p)
        data, raw = _read_yaml(p, probs)
        if raw is not None:
            sources.append(p)
            digests.append(_sha256(raw))
        if data is None:
            continue
        try:
            rf = RuleFile.model_validate(data)
        except ValidationError as exc:
            probs.extend(_validation_problems(p, exc, "Match the RuleSpec schema (closed predicate union, "
                                                      "unknown keys rejected)."))
            continue
        for r in rf.rules:
            origins.setdefault(r.id, str(p))
            rules.append(r)
    extractors = _load_extractors(extractor_paths, probs)
    try:
        bundle = build_bundle(rules, extractors, templates=templates, origins=origins,
                              sources=sources, source_digests=digests)
    except RuleLoadError as exc:
        probs.extend(exc.details)
        bundle = None
    if probs or bundle is None:
        raise RuleLoadError([str(x) for x in probs], probs)
    return bundle


class _TemplateFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: int = 1
    templates: list[TemplateSpec]


def load_templates(path: Path) -> Mapping[str, TemplateSpec]:
    """Load a template catalog file (``schema_version`` + ``templates`` list)."""
    probs: list[Problem] = []
    data, _ = _read_yaml(Path(path), probs)
    if probs:
        raise RuleLoadError([str(x) for x in probs], probs)
    return _parse_templates(data, str(path))


def _parse_templates(data: Any, where: str) -> Mapping[str, TemplateSpec]:
    try:
        tf = _TemplateFile.model_validate(data)
    except ValidationError as exc:
        probs = _validation_problems(Path(where), exc, "Match the TemplateSpec schema.")
        raise RuleLoadError([str(x) for x in probs], probs) from None
    out: dict[str, TemplateSpec] = {}
    for t in tf.templates:
        if t.id in out:
            prob = Problem(where, f"duplicate template id {t.id!r}", "Keep one entry per id.")
            raise RuleLoadError([str(prob)], [prob])
        out[t.id] = t
    return MappingProxyType(out)


@lru_cache(maxsize=1)
def default_templates() -> Mapping[str, TemplateSpec]:
    """The trusted remediation catalog shipped as ``ci_lab/rules/templates.yaml`` (B3)."""
    pkg, name = TEMPLATES_RESOURCE
    text = resources.files(pkg).joinpath(name).read_text(encoding="utf-8")
    return _parse_templates(yaml.load(text, Loader=_UniqueKeyLoader), f"{pkg}/{name}")


# ---------------------------------------------------------------- last-known-good (N1)


def _guard_files(guards_dir: Path) -> list[Path]:
    d = Path(guards_dir)
    if not d.is_dir():
        return []
    return sorted([*d.glob("*.yaml"), *d.glob("*.yml")], key=lambda p: p.name)


def write_lkg(guards_dir: Path, bundle: Bundle, lock: Path | None = None) -> Path:
    """Copy the bundle's validated rule files into ``<guards_dir>/.lkg/`` and pin them in the
    lock (default ``<guards_dir>/BUNDLE.lock``): ``{"digest", "files": {name: sha256}}``.

    Refuses (RuleLoadError) if a source changed since it was validated or names collide.
    Returns the lock path. Used by publish; never by arms.
    """
    guards_dir = Path(guards_dir)
    lock = Path(lock) if lock is not None else guards_dir / LOCK_NAME
    blobs: dict[str, bytes] = {}
    probs: list[Problem] = []
    for src, want in zip(bundle.sources, bundle.source_digests, strict=True):
        raw = Path(src).read_bytes()
        if _sha256(raw) != want:
            probs.append(Problem(str(src), "file changed since the bundle was validated",
                                 "Reload the bundle, then write the LKG again."))
        if src.name in blobs:
            probs.append(Problem(str(src), f"duplicate file name {src.name!r}", "Give rule files unique names."))
        blobs[src.name] = raw
    if probs:
        raise RuleLoadError([str(x) for x in probs], probs)
    lkg = guards_dir / LKG_DIRNAME
    lkg.mkdir(parents=True, exist_ok=True)
    for old in [*lkg.glob("*.yaml"), *lkg.glob("*.yml")]:
        if old.name not in blobs:
            old.unlink()
    for name, raw in blobs.items():
        tmp = lkg / (name + ".tmp")
        tmp.write_bytes(raw)
        tmp.replace(lkg / name)
    body = {"digest": bundle.digest, "files": {n: _sha256(b) for n, b in sorted(blobs.items())}}
    lock.parent.mkdir(parents=True, exist_ok=True)
    tmp = lock.with_name(lock.name + ".tmp")
    tmp.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(lock)
    return lock


def _load_lkg(guards_dir: Path, lock: Path, extractor_paths: Sequence[Path],
              templates: Mapping[str, TemplateSpec] | None) -> Bundle:
    def fail(msg: str) -> RuleLoadError:
        prob = Problem(str(lock), f"LKG unusable: {msg}", "Restore harness/guards or republish a valid bundle.")
        return RuleLoadError([str(prob)], [prob])

    try:
        meta = json.loads(Path(lock).read_text(encoding="utf-8"))
        digest, files = meta["digest"], meta["files"]
        if not isinstance(digest, str) or not isinstance(files, dict):
            raise TypeError
    except (OSError, ValueError, KeyError, TypeError):
        raise fail("lock missing or malformed") from None
    lkg = Path(guards_dir) / LKG_DIRNAME
    paths = []
    for name, sha in sorted(files.items()):
        if not isinstance(name, str) or Path(name).name != name or name in ("", ".", ".."):
            raise fail(f"bad file name {name!r}")
        p = lkg / name
        try:
            raw = p.read_bytes()
        except OSError:
            raise fail(f"missing {name}") from None
        if _sha256(raw) != sha:
            raise fail(f"{name} does not match the lock")
        paths.append(p)
    bundle = load_bundle(paths, extractor_paths, templates=templates)
    if bundle.digest != digest:
        raise fail("bundle digest does not match the lock")
    return bundle


def load_with_lkg(
    guards_dir: Path,
    lock: Path,
    extractor_paths: Sequence[Path] = (),
    *,
    templates: Mapping[str, TemplateSpec] | None = None,
) -> tuple[Bundle, bool]:
    """Load ``<guards_dir>/*.yaml``; on failure fall back to the digest-pinned LKG copy (N1).

    Returns ``(bundle, degraded)``; ``degraded`` is True when the LKG copy was used (callers emit
    ``ci.guard.degraded``). Raises :class:`RuleLoadError` with both failures if neither loads.
    """
    try:
        return load_bundle(_guard_files(guards_dir), extractor_paths, templates=templates), False
    except RuleLoadError as exc:
        current = exc
    try:
        return _load_lkg(Path(guards_dir), Path(lock), extractor_paths, templates), True
    except RuleLoadError as exc:
        details = [*current.details, *exc.details]
        raise RuleLoadError([str(x) for x in details], details) from None
