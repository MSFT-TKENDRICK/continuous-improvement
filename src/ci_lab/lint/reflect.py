"""``ci-lab reflect`` — mine *local* Copilot CLI session transcripts for repeated corrections and
propose lint rules (design §13.1 R5, §13.4; privacy rules §13.6 B8).

Privacy contract (B8):
* opt-in only (``--i-consent-local-mining``); reads local files, sends nothing anywhere;
* only sessions whose cwd/gitRoot is under this repo (repo-origin provenance) are read;
* each event's text is inspected in memory by deterministic detectors and dropped immediately —
  events that look like they carry secrets/PII are dropped whole and only counted;
* outputs contain only a closed vocabulary: template ids and rule text from the trusted catalog
  (``templates.yaml``), existing lint rule ids, ``dir/*.ext`` globs of directories that exist in the
  repo, tool names matching a strict pattern, and counts. No excerpts, args, messages or hashes;
* a lesson is proposed only when ≥ ``min_sessions`` distinct sessions agree (convergence), and
  proposed rules start at ``severity: warn`` (shadow) for human review via a draft PR.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import re2
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ci_lab.lint.spec import (
    LintRuleFile,
    RuleLoadError,
    check_re2,
    load_rules,
    rule_files,
)
from ci_lab.rulespec import CorrectionRecord

TEMPLATES_PATH = Path(__file__).with_name("templates.yaml")
DEFAULT_OUT = Path("artifacts/reflect/proposals.yaml")
LINT_TOOL = "ci-lab-lint"
MAX_EVENT_TEXT = 200_000


class ReflectError(RuntimeError):
    pass


def _rx(p: str):
    return re2.compile(p)


CORRECTION_RE = _rx(r"(?i)\b(?:don'?t|do not|never|stop|again|avoid|no more|quit|instead of)\b")
LINT_HEADER_RE = _rx(r"\[LINT\]\[(?:ERROR|WARN)\] ([^\s:]{1,300}):\d+")
LINT_VIOLATION_RE = _rx(r"Violation: \[([a-z][a-z0-9_.-]{2,63})\]")
FAIL_RE = _rx(r"(?i)(?:exit code|exited with code|exit status):? [1-9]|\bFAILED\b"
              r"|Traceback \(most recent call last\)|\[LINT\] Failed")
TOOL_NAME_RE = _rx(r"^[a-z_][a-z0-9_-]{0,31}$")
EDIT_TOOL_RE = _rx(r"(?i)edit|create|write|patch|replace|insert")
SAFE_GLOB_RE = _rx(r"^[A-Za-z0-9_.\-/]*\*(?:\.[A-Za-z0-9]{1,8})?$")
SECRET_RES = [_rx(p) for p in (
    r"\bgh[pousr]_[A-Za-z0-9]{20,}",
    r"\bgithub_pat_[A-Za-z0-9_]{20,}",
    r"\bAKIA[0-9A-Z]{16}\b",
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.",
    r"\bxox[abprs]-[A-Za-z0-9-]{10,}",
    r"\bsk-[A-Za-z0-9_-]{20,}",
    r"(?i)\b(?:password|passwd|secret|api[_-]?key|access[_-]?token)\s*[:=]\s*\S{6,}",
)]
PII_RES = [_rx(p) for p in (
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    r"\b\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}\b",
    r"\b\d{3}-\d{2}-\d{4}\b",
)]


def has_secret(text: str) -> bool:
    return any(r.search(text) for r in SECRET_RES)


def has_pii(text: str) -> bool:
    return has_secret(text) or any(r.search(text) for r in PII_RES)


# ---------------------------------------------------------------- templates


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Template(_M):
    id: str = Field(pattern=r"^[a-z][a-z0-9-]{2,40}$")
    triggers: list[str] = Field(min_length=1)
    rule: dict[str, Any]

    @field_validator("triggers")
    @classmethod
    def _t(cls, v: list[str]) -> list[str]:
        return [check_re2(p) for p in v]


class TemplateFile(_M):
    schema_version: int = 1
    templates: list[Template]


def load_templates(path: Path = TEMPLATES_PATH) -> dict[str, Template]:
    try:
        tf = TemplateFile.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
        for t in tf.templates:  # every skeleton must be a valid lint rule
            LintRuleFile.model_validate({"rules": [{"id": f"lesson.{t.id}", **t.rule}]})
    except (OSError, yaml.YAMLError, ValidationError) as e:
        raise ReflectError(f"invalid template catalog {path.name}: {type(e).__name__}") from None
    return {t.id: t for t in tf.templates}


# ---------------------------------------------------------------- sessions


def _norm(p: str | Path) -> str:
    return os.path.normcase(os.path.normpath(os.path.abspath(str(p))))


def is_under(path: str | Path, root: Path) -> bool:
    p, r = _norm(path), _norm(root)
    return p == r or p.startswith(r.rstrip(os.sep) + os.sep)


def _iter_events(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if len(line) > MAX_EVENT_TEXT * 2:
                continue
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if isinstance(ev, dict):
                yield ev


def session_cwd(session_dir: Path) -> Path | None:
    """Session working directory from the first ``session.start`` event, else ``workspace.yaml``."""
    events = session_dir / "events.jsonl"
    if events.is_file():
        for i, ev in enumerate(_iter_events(events)):
            if ev.get("type") == "session.start":
                ctx = (ev.get("data") or {}).get("context") or {}
                for key in ("cwd", "gitRoot"):
                    if isinstance(ctx.get(key), str) and ctx[key]:
                        return Path(ctx[key])
                break
            if i > 50:
                break
    ws = session_dir / "workspace.yaml"
    if ws.is_file():
        try:
            doc = yaml.safe_load(ws.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            return None
        if isinstance(doc, dict) and isinstance(doc.get("cwd"), str):
            return Path(doc["cwd"])
    return None


def file_glob(path: str, cwd: Path, root: Path) -> str | None:
    """Generalize a file path to ``dir/*.ext`` relative to the repo; None if outside or unknown."""
    if not isinstance(path, str) or not path or len(path) > 400 or "\0" in path:
        return None
    p = Path(path)
    p = p if p.is_absolute() else Path(cwd) / p
    if not is_under(p, root):
        return None
    rel = Path(os.path.relpath(os.path.normpath(os.path.abspath(p)), os.path.abspath(root)))
    parent = rel.parent
    if not (Path(root) / parent).is_dir():
        return None
    ext = rel.suffix if re2.fullmatch(r"\.[A-Za-z0-9]{1,8}", rel.suffix or "") else ""
    glob = ("" if str(parent) == "." else parent.as_posix() + "/") + "*" + ext
    return glob if SAFE_GLOB_RE.fullmatch(glob) else None


def _tool(name: Any) -> str:
    s = str(name or "").lower()
    return s if TOOL_NAME_RE.fullmatch(s) else "other"


def split_commands(command: str) -> list[list[str]]:
    """Quote-aware split of a shell line into token lists (POSIX sh / PowerShell separators)."""
    cmds: list[list[str]] = []
    toks: list[str] = []
    cur, has, quote = [], False, None
    for c in command:
        if quote:
            if c == quote:
                quote = None
            else:
                cur.append(c)
            continue
        if c in "\"'":
            quote, has = c, True
        elif c in ";\n\r|&(){}":
            if has:
                toks.append("".join(cur))
            cur, has = [], False
            if toks:
                cmds.append(toks)
            toks = []
        elif c in " \t":
            if has:
                toks.append("".join(cur))
            cur, has = [], False
        else:
            cur.append(c)
            has = True
    if has:
        toks.append("".join(cur))
    if toks:
        cmds.append(toks)
    return cmds


def revert_targets(command: str) -> tuple[list[str], bool]:
    """(paths restored by git checkout/restore, whole-tree revert?) for one shell command line."""
    paths: list[str] = []
    whole = False
    for toks in split_commands(command):
        try:
            i = next(k for k, t in enumerate(toks) if re2.fullmatch(r"(?i)(?:.*[\\/])?git(?:\.exe)?", t))
        except StopIteration:
            continue
        rest = toks[i + 1:]
        while rest and rest[0].startswith("-"):
            rest = rest[2:] if rest[0] in ("-C", "-c") else rest[1:]
        if not rest:
            continue
        sub, args = rest[0], rest[1:]
        if (sub == "reset" and "--hard" in args) or sub == "revert":
            whole = True
        elif sub == "checkout" and "--" in args:
            paths += [a for a in args[args.index("--") + 1:] if a]
        elif sub == "restore" and not (("--staged" in args or "-S" in args) and "--worktree" not in args
                                       and "-W" not in args):
            skip = False
            for a in args:
                if skip:
                    skip = False
                elif a in ("--source", "-s"):
                    skip = True
                elif not a.startswith("-"):
                    paths.append(a)
    return paths, whole


@dataclass
class SessionScan:
    records: list[CorrectionRecord] = field(default_factory=list)
    dropped: int = 0


def scan_session(events_path: Path, *, root: Path, cwd: Path, known_rules: set[str],
                 templates: dict[str, Template]) -> SessionScan:
    """Deterministic detectors over one session; raw text never leaves this function."""
    trig = {tid: [_rx(p) for p in t.triggers] for tid, t in templates.items()}
    out = SessionScan()
    edited: dict[str, str] = {}  # glob -> edit tool
    last_rule: dict[str, str] = {}  # glob -> last lint rule that failed there
    lint_fails: Counter[tuple[str, str | None]] = Counter()
    reverts: Counter[tuple[str | None, str]] = Counter()
    fixes: Counter[tuple[str, str]] = Counter()
    corrections: Counter[str] = Counter()
    fail_pending = False
    for ev in _iter_events(events_path):
        typ = ev.get("type")
        data = ev.get("data")
        if not isinstance(data, dict):
            continue
        if typ == "tool.execution_start":
            name = _tool(data.get("toolName"))
            args = data.get("arguments")
            if not isinstance(args, dict):
                continue
            cmd = args.get("command")
            if isinstance(cmd, str):
                if len(cmd) > MAX_EVENT_TEXT or has_pii(cmd):
                    out.dropped += 1
                    continue
                paths, whole = revert_targets(cmd)
                del cmd
                if whole and edited:
                    reverts[(None, Counter(edited.values()).most_common(1)[0][0])] += 1
                for p in paths:
                    g = file_glob(p, cwd, root)
                    if g and g in edited:
                        reverts[(g, edited[g])] += 1
            elif EDIT_TOOL_RE.search(name) and isinstance(args.get("path"), str):
                g = file_glob(args["path"], cwd, root)
                if g:
                    edited[g] = name
                    if fail_pending:
                        fixes[(g, name)] += 1
                        fail_pending = False
        elif typ == "tool.execution_complete":
            result = data.get("result")
            text = ""
            if isinstance(result, dict):
                text = result.get("content") or result.get("detailedContent") or ""
            elif isinstance(result, str):
                text = result
            if not isinstance(text, str) or len(text) > MAX_EVENT_TEXT:
                text = ""
            if text and has_secret(text):
                out.dropped += 1
                text = ""
            if data.get("success") is False or (text and FAIL_RE.search(text)):
                fail_pending = True
            path_glob: str | None = None
            for line in text.splitlines() if "[LINT]" in text else ():
                if m := LINT_HEADER_RE.search(line):
                    path_glob = file_glob(m.group(1), root, root)
                elif (m := LINT_VIOLATION_RE.search(line)) and m.group(1) in known_rules:
                    lint_fails[(m.group(1), path_glob)] += 1
                    if path_glob:
                        last_rule[path_glob] = m.group(1)
            del text
        elif typ == "user.message":
            msg = data.get("content")
            if not isinstance(msg, str) or len(msg) > MAX_EVENT_TEXT:
                continue
            if has_pii(msg):
                out.dropped += 1
                continue
            if CORRECTION_RE.search(msg):
                for tid, rxs in trig.items():
                    if any(r.search(msg) for r in rxs):
                        corrections[tid] += 1
            del msg
    recs = out.records
    recs += [CorrectionRecord(kind="lint_fail", tool=LINT_TOOL, rule_hint=r, file_glob=g, count=n)
             for (r, g), n in sorted(lint_fails.items(), key=str)]
    recs += [CorrectionRecord(kind="revert", tool=t, rule_hint=last_rule.get(g) if g else None, file_glob=g,
                              count=n) for (g, t), n in sorted(reverts.items(), key=str)]
    recs += [CorrectionRecord(kind="repeated_fix", tool=t, rule_hint=last_rule.get(g), file_glob=g, count=n)
             for (g, t), n in sorted(fixes.items()) if n >= 2]
    recs += [CorrectionRecord(kind="user_correction", rule_hint=tid, count=n)
             for tid, n in sorted(corrections.items())]
    return out


# ---------------------------------------------------------------- aggregate + propose


@dataclass
class Lesson:
    record: CorrectionRecord  # aggregated (count = total across sessions)
    sessions: int
    proposal: str | None = None


def aggregate(per_session: Sequence[Sequence[CorrectionRecord]]) -> list[Lesson]:
    counts: Counter[tuple] = Counter()
    sessions: defaultdict[tuple, set[int]] = defaultdict(set)
    for i, recs in enumerate(per_session):
        for r in recs:
            k = (r.kind, r.tool, r.rule_hint, r.file_glob)
            counts[k] += r.count
            sessions[k].add(i)
    return [Lesson(CorrectionRecord(kind=k[0], tool=k[1], rule_hint=k[2], file_glob=k[3], count=counts[k]),
                   len(sessions[k])) for k in sorted(counts, key=lambda k: tuple(str(x) for x in k))]


def _slug(glob: str) -> str:
    s = re2.sub(r"[^a-z0-9]+", "-", glob.lower()).strip("-")
    return s or "root"


def propose(lessons: Sequence[Lesson], templates: dict[str, Template], known_rules: set[str], *,
            min_sessions: int) -> tuple[list[Lesson], list[dict]]:
    """Converged lessons (≥ min_sessions) → proposed warn-level lint rules from trusted templates."""
    converged = [ls for ls in lessons if ls.sessions >= min_sessions]
    rules: dict[str, dict] = {}
    for ls in converged:
        hint, glob = ls.record.rule_hint, ls.record.file_glob
        if hint in templates:
            t = templates[hint]
            rid = f"lesson.{hint}" + (f".{_slug(glob)}" if glob else "")
            rid = rid[:64].rstrip("-.")
            ls.proposal = rid
            if rid in rules:
                continue
            body = {"id": rid, **t.rule, "severity": "warn",
                    "see": f"ci-lab reflect (design §13.6 B8): {ls.record.kind} x{ls.record.count} "
                           f"across {ls.sessions} sessions"}
            if glob:
                body["include"] = [glob]
            rules[rid] = body
        elif hint in known_rules:
            ls.proposal = f"existing:{hint}"
    if rules:
        LintRuleFile.model_validate({"rules": list(rules.values())})
    return converged, list(rules.values())


# ---------------------------------------------------------------- draft PR (git plumbing)


def _git(root: Path, *args: str, env: dict | None = None, data: bytes | None = None) -> str:
    r = subprocess.run(["git", "-C", str(root), *args], input=data, capture_output=True, env=env,
                       check=False, timeout=60)
    if r.returncode != 0:
        raise ReflectError(f"git {args[0]} failed ({r.returncode})")
    return r.stdout.decode("utf-8", "replace").strip()


def draft_branch(root: Path, rules_yaml: str, today: str, body_path: Path) -> tuple[str, str]:
    """Commit ``lint/rules/lessons-<date>.yaml`` on a new local branch without touching the working
    tree or the user's index. Returns (branch, gh command to run manually)."""
    branch = f"lessons/reflect-{today}"
    exists = subprocess.run(["git", "-C", str(root), "rev-parse", "--verify", "--quiet", f"refs/heads/{branch}"],
                            capture_output=True, check=False, timeout=60)
    if exists.returncode == 0:
        raise ReflectError(f"branch {branch} already exists")
    base = _git(root, "rev-parse", "--abbrev-ref", "HEAD")
    idx = Path(root) / "artifacts" / "reflect" / f".index-{os.getpid()}"
    idx.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "GIT_INDEX_FILE": str(idx)}
    try:
        _git(root, "read-tree", "HEAD", env=env)
        blob = _git(root, "hash-object", "-w", "--stdin", data=rules_yaml.encode("utf-8"))
        _git(root, "update-index", "--add", "--cacheinfo", f"100644,{blob},lint/rules/lessons-{today}.yaml", env=env)
        tree = _git(root, "write-tree", env=env)
        msg = f"lessons: proposed lint rules from ci-lab reflect ({today})\n\nShadow (severity: warn); review before promoting.\n"
        commit = _git(root, "commit-tree", tree, "-p", "HEAD", data=msg.encode("utf-8"))
        _git(root, "branch", branch, commit)
    finally:
        idx.unlink(missing_ok=True)
    rel_body = os.path.relpath(body_path, root).replace(os.sep, "/")
    gh = (f"git push -u origin {branch} && gh pr create --draft --base {base} --head {branch} "
          f'--title "lessons: lint rules from ci-lab reflect {today}" --body-file {rel_body}')
    return branch, gh


# ---------------------------------------------------------------- entry point


def _record(r: CorrectionRecord) -> dict:
    return r.model_dump(exclude_none=True)


def reflect_main(*, root: Path, sessions_dir: Path, out: Path | None = None, min_sessions: int = 2,
                 keep: bool = False, draft_pr: bool = False, today: str | None = None,
                 rule_paths: Sequence[Path] | None = None) -> int:
    root = Path(root).resolve()
    sessions_dir = Path(sessions_dir)
    if min_sessions < 2:
        raise ReflectError("--min-sessions must be >= 2 (convergence across sessions, B8)")
    if not sessions_dir.is_dir():
        raise ReflectError(f"sessions dir not found: {sessions_dir}")
    try:
        known_rules = {r.id for r in load_rules(rule_paths if rule_paths is not None else rule_files(root))}
    except RuleLoadError as e:
        raise ReflectError(f"lint rules do not load ({len(e.errors)} error(s))") from None
    templates = load_templates()
    today = today or dt.datetime.now(dt.UTC).date().isoformat()

    scanned = in_repo = dropped = 0
    per_session: list[list[CorrectionRecord]] = []
    for sdir in sorted(p for p in sessions_dir.iterdir() if p.is_dir()):
        events = sdir / "events.jsonl"
        if not events.is_file():
            continue
        scanned += 1
        cwd = session_cwd(sdir)
        if cwd is None or not is_under(cwd, root):
            continue
        in_repo += 1
        scan = scan_session(events, root=root, cwd=cwd, known_rules=known_rules, templates=templates)
        dropped += scan.dropped
        per_session.append(scan.records)

    lessons = aggregate(per_session)
    converged, rules = propose(lessons, templates, known_rules, min_sessions=min_sessions)
    out = Path(out) if out else root / DEFAULT_OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "schema_version": 1,
        "generated": today,
        "source": "copilot-sessions",
        "sessions": {"scanned": scanned, "in_repo": in_repo},
        "dropped_sensitive_events": dropped,
        "min_sessions": min_sessions,
        "lessons": [{**_record(ls.record), "sessions": ls.sessions, "proposal": ls.proposal} for ls in converged],
        "below_convergence": len(lessons) - len(converged),
        "proposed_rules": {"schema_version": 1, "rules": rules},
    }
    out.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")
    if keep:
        with (out.parent / "records.jsonl").open("w", encoding="utf-8") as f:
            for i, recs in enumerate(per_session):
                for r in recs:
                    f.write(json.dumps({"session": i, **_record(r)}, sort_keys=True) + "\n")
    print(f"reflect: {in_repo}/{scanned} session(s) in repo; {len(lessons)} lesson(s), {len(converged)} converged "
          f"(>= {min_sessions} sessions); {len(rules)} rule(s) proposed; {dropped} sensitive event(s) dropped")
    print(f"reflect: wrote {out}")
    if draft_pr:
        if not rules:
            print("reflect: nothing to propose; no branch created")
            return 0
        rules_yaml = ("# Proposed by `ci-lab reflect` (design §13.6 B8). Shadow rules: review, then promote to error.\n"
                      + yaml.safe_dump({"schema_version": 1, "rules": rules}, sort_keys=False, allow_unicode=True))
        body = out.parent / "pr-body.md"
        body.write_text(_pr_body(converged, rules, in_repo, today), encoding="utf-8")
        branch, gh = draft_branch(root, rules_yaml, today, body)
        print(f"reflect: created local branch {branch} (working tree untouched). To open the draft PR run:")
        print(f"  {gh}")
    return 0


def _pr_body(lessons: Sequence[Lesson], rules: Sequence[dict], sessions: int, today: str) -> str:
    lines = [f"Lessons mined by `ci-lab reflect` on {today} from {sessions} local session(s) in this repo.",
             "", "Aggregate counts only (B8): no transcript text is included.", "",
             "| kind | rule_hint | file_glob | count | sessions | proposal |", "|---|---|---|---|---|---|"]
    lines += [f"| {ls.record.kind} | {ls.record.rule_hint or ''} | {ls.record.file_glob or ''} | "
              f"{ls.record.count} | {ls.sessions} | {ls.proposal or ''} |" for ls in lessons]
    lines += ["", f"{len(rules)} rule(s) proposed at `severity: warn`; promote to `error` once clean."]
    return "\n".join(lines) + "\n"
