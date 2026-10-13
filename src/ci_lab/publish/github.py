"""GitHub publishing of RRSI rounds via the ``gh``/``git`` CLIs (design-oes-rrsi-v1 §3, C6).

* Accepted arm  -> pushed ``exp/<eid>/<arm>`` branch + PR whose base is the previous
  accepted layer (bottom base = ``main``); registered in a native stack at the 2nd
  acceptance (``POST repos/{o}/{r}/stacks``, needs >= 2 PRs) and appended after
  (``POST repos/{o}/{r}/stacks/{n}/add``).
* Losers        -> archive tags ``exp-archive/<eid>/<arm>`` (no PRs).
* Every mutation: snapshot -> revalidate -> mutate -> verify, run through the
  outbox under a deterministic op id with the snapshot as reconcile probe.
* Every dynamic value is validated before it reaches an argv list (never a shell).
* ``dry_run=True`` records the intended mutating calls in :attr:`calls` and returns
  synthetic results (queries report "absent"); dry-run op ids are namespaced so
  they never satisfy a real run.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ci_lab.contracts import Outbox, arm_branch, op_id

PR_RE = re.compile(r"^[0-9]{1,10}$")
SHA_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
REF_RE = re.compile(r"^[A-Za-z0-9._/-]+$")
NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_PR_URL_RE = re.compile(r"/pull/([0-9]{1,10})\s*$")
MAX_TITLE = 256
MAX_BODY = 60_000

Runner = Callable[[Sequence[str]], str]


class PublishError(RuntimeError):
    pass


class CommandError(PublishError):
    def __init__(self, args: Sequence[str], returncode: int, stderr: str) -> None:
        super().__init__(f"{' '.join(args[:3])} ... exited {returncode}: {stderr.strip()[-400:]}")
        self.returncode = returncode
        self.stderr = stderr


# ------------------------------------------------------------------ validation

def validate_pr_number(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"bad PR number {value!r}")
    text = str(value)
    if not PR_RE.fullmatch(text) or int(text) <= 0:
        raise ValueError(f"bad PR number {value!r}")
    return int(text)


def validate_sha(value: Any) -> str:
    if not isinstance(value, str) or not SHA_RE.fullmatch(value):
        raise ValueError(f"bad commit sha {value!r}")
    return value


def validate_name(value: Any, what: str = "name") -> str:
    if not isinstance(value, str) or not NAME_RE.fullmatch(value) or value in (".", "..") \
            or value.startswith("-") or len(value) > 100:
        raise ValueError(f"bad {what} {value!r}")
    return value


def validate_repo(value: Any) -> tuple[str, str]:
    if not isinstance(value, str) or value.count("/") != 1:
        raise ValueError(f"bad owner/repo {value!r}")
    owner, repo = value.split("/")
    return validate_name(owner, "owner"), validate_name(repo, "repo")


def _git_check_ref_format(full_ref: str) -> bool:
    git = shutil.which("git")
    if git is None:
        return True  # python rules below already enforce the git grammar subset we allow
    proc = subprocess.run([git, "check-ref-format", full_ref], capture_output=True, check=False, timeout=30)
    return proc.returncode == 0


def validate_ref(value: Any, *, namespace: str = "heads") -> str:
    """Branch/tag short name: ``^[A-Za-z0-9._/-]+$`` + git ref grammar + ``git check-ref-format``."""
    if not isinstance(value, str) or not REF_RE.fullmatch(value) or len(value) > 200:
        raise ValueError(f"bad ref {value!r}")
    parts = value.split("/")
    if value.startswith("-") or ".." in value or "//" in value or value.endswith((".", "/", ".lock")) \
            or any(not p or p.startswith(".") or p.endswith(".lock") for p in parts) or value == "HEAD":
        raise ValueError(f"bad ref {value!r}")
    if not _git_check_ref_format(f"refs/{namespace}/{value}"):
        raise ValueError(f"bad ref {value!r} (git check-ref-format)")
    return value


def validate_text(value: Any, max_len: int, what: str) -> str:
    if not isinstance(value, str) or len(value) > max_len or "\x00" in value or \
            any(ord(c) < 32 and c not in "\n\t\r" for c in value):
        raise ValueError(f"bad {what}")
    return value


def archive_tag_name(eid: str, arm: str) -> str:
    branch = arm_branch(eid, arm)  # validates eid/arm grammar
    return "exp-archive/" + branch.removeprefix("exp/")


def subprocess_runner(cwd: Path | None = None, timeout: float = 300) -> Runner:
    def run(args: Sequence[str]) -> str:
        proc = subprocess.run(list(args), cwd=cwd, capture_output=True, text=True, timeout=timeout,
                              check=False, shell=False)
        if proc.returncode != 0:
            raise CommandError(args, proc.returncode, proc.stderr)
        return proc.stdout

    return run


def _parse_stack(payload: Any) -> dict[str, Any] | None:
    """Normalise a stacks API payload to ``{"number": int, "pull_requests": [int]}``."""
    if isinstance(payload, Mapping) and isinstance(payload.get("stacks"), list):
        payload = payload["stacks"]
    if isinstance(payload, list):
        payload = payload[0] if payload else None
    if not isinstance(payload, Mapping):
        return None
    number = payload.get("number", payload.get("id"))
    prs = []
    for item in payload.get("pull_requests") or payload.get("pullRequests") or []:
        prs.append(validate_pr_number(item.get("number") if isinstance(item, Mapping) else item))
    return {"number": validate_pr_number(number), "pull_requests": prs}


def _synthetic_number(*parts: Any) -> int:
    return 100_000 + int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16) % 900_000


# ------------------------------------------------------------------ publisher

class GitHubPublisher:
    def __init__(self, repo: str, *, outbox: Outbox, runner: Runner | None = None, dry_run: bool = False,
                 remote: str = "origin", base: str = "main", draft: bool = True, git_cwd: Path | None = None,
                 journal: Path | None = None) -> None:
        self.owner, self.repo = validate_repo(repo)
        self.remote = validate_name(remote, "remote")
        self.base = validate_ref(base)
        self.outbox = outbox
        self.runner = runner or subprocess_runner(git_cwd)
        self.dry_run = dry_run
        self.draft = draft
        self.journal = journal
        self.calls: list[list[str]] = []
        self.queries: list[list[str]] = []

    @property
    def slug(self) -> str:
        return f"{self.owner}/{self.repo}"

    # ---- plumbing
    def _op(self, *parts: Any) -> str:
        return op_id("dry-run" if self.dry_run else "publish", self.slug, *parts)

    def _mutate(self, args: Sequence[str]) -> str:
        argv = [str(a) for a in args]
        self.calls.append(argv)
        if self.journal is not None:
            self.journal.parent.mkdir(parents=True, exist_ok=True)
            with self.journal.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"dry_run": self.dry_run, "argv": argv}) + "\n")
        return "" if self.dry_run else self.runner(argv)

    def _query(self, args: Sequence[str]) -> str | None:
        argv = [str(a) for a in args]
        self.queries.append(argv)
        return None if self.dry_run else self.runner(argv)

    def _guarded(self, op: str, *, snapshot: Callable[[], Any | None], revalidate: Callable[[], None],
                 mutate: Callable[[], Any], verify: Callable[[Any], bool]) -> Any:
        def act() -> Any:
            found = snapshot()
            if found is not None:
                return found
            revalidate()
            result = mutate()
            if not self.dry_run and not verify(result):
                raise PublishError(f"post-mutation verification failed for op {op}")
            return result

        return self.outbox.run_once(op, act, reconcile=snapshot)

    # ---- git refs
    def remote_sha(self, full_ref: str) -> str | None:
        out = self._query(["git", "ls-remote", self.remote, full_ref])
        for line in (out or "").splitlines():
            sha, _, ref = line.partition("\t")
            if ref.strip() == full_ref:
                return validate_sha(sha.strip())
        return None

    def _push_ref(self, kind: str, full_ref: str, sha: str) -> dict[str, Any]:
        def snapshot() -> dict[str, Any] | None:
            current = self.remote_sha(full_ref)
            if current is None:
                return None
            if current != sha:
                raise PublishError(f"{full_ref} exists at another commit; refusing to move it")
            return {"ref": full_ref, "sha": sha}

        return self._guarded(
            self._op(kind, full_ref, sha), snapshot=snapshot, revalidate=lambda: None,
            mutate=lambda: (self._mutate(["git", "push", self.remote, f"{sha}:{full_ref}"]),
                            {"ref": full_ref, "sha": sha})[1],
            verify=lambda _r: self.remote_sha(full_ref) == sha)

    def push_branch(self, branch: str, sha: str) -> dict[str, Any]:
        return self._push_ref("push", f"refs/heads/{validate_ref(branch)}", validate_sha(sha))

    def archive_tag(self, eid: str, arm: str, sha: str) -> dict[str, Any]:
        tag = validate_ref(archive_tag_name(eid, arm), namespace="tags")
        return self._push_ref("tag", f"refs/tags/{tag}", validate_sha(sha))

    # ---- pull requests
    def find_pr(self, head: str) -> dict[str, Any] | None:
        validate_ref(head)
        out = self._query(["gh", "pr", "list", "--repo", self.slug, "--head", head, "--state", "all",
                           "--json", "number,headRefName,baseRefName,state,isDraft,url", "--limit", "20"])
        if not out:
            return None
        rows = [r for r in json.loads(out) if r.get("headRefName") == head]
        rows.sort(key=lambda r: r.get("state") != "OPEN")
        if not rows:
            return None
        r = rows[0]
        return {"number": validate_pr_number(r["number"]), "head": head, "base": r.get("baseRefName"),
                "state": r.get("state"), "draft": r.get("isDraft"), "url": r.get("url")}

    def open_pr(self, head: str, base: str, title: str, body: str, draft: bool = True) -> dict[str, Any]:
        validate_ref(head)
        validate_ref(base)
        validate_text(title, MAX_TITLE, "title")
        validate_text(body, MAX_BODY, "body")

        def snapshot() -> dict[str, Any] | None:
            pr = self.find_pr(head)
            if pr is None:
                return None
            if pr["state"] != "OPEN" or pr["base"] != base:
                raise PublishError(f"existing PR #{pr['number']} for {head} is {pr['state']} on {pr['base']}")
            return pr

        def revalidate() -> None:
            if self.dry_run:
                return
            for ref in (head, base):
                if self.remote_sha(f"refs/heads/{ref}") is None:
                    raise PublishError(f"branch {ref} is not on {self.remote}")

        def mutate() -> dict[str, Any]:
            args = ["gh", "pr", "create", "--repo", self.slug, "--head", head, "--base", base,
                    f"--title={title}", f"--body={body}"]
            if draft:
                args.append("--draft")
            out = self._mutate(args)
            if self.dry_run:
                number = _synthetic_number(self.slug, head)
            else:
                m = _PR_URL_RE.search(out.strip())
                if not m:
                    raise PublishError("could not parse PR url from gh output")
                number = validate_pr_number(m.group(1))
            return {"number": number, "head": head, "base": base, "state": "OPEN", "draft": draft}

        def verify(result: Mapping[str, Any]) -> bool:
            pr = self.find_pr(head)
            return bool(pr and pr["number"] == result["number"] and pr["base"] == base)

        return self._guarded(self._op("pr", head, base), snapshot=snapshot, revalidate=revalidate,
                             mutate=mutate, verify=verify)

    def pr_view(self, number: int) -> dict[str, Any] | None:
        n = validate_pr_number(number)
        out = self._query(["gh", "pr", "view", str(n), "--repo", self.slug, "--json", "number,state,isDraft"])
        return json.loads(out) if out else None

    # ---- native stacks
    def stack_of(self, pr: int) -> dict[str, Any] | None:
        n = validate_pr_number(pr)
        try:
            out = self._query(["gh", "api", f"repos/{self.slug}/stacks?pull_request={n}"])
        except CommandError as exc:
            if "404" in exc.stderr or "Not Found" in exc.stderr:
                return None
            raise
        return _parse_stack(json.loads(out)) if out else None

    def create_stack(self, prs: Sequence[int]) -> dict[str, Any]:
        numbers = [validate_pr_number(p) for p in prs]
        if len(numbers) < 2 or len(set(numbers)) != len(numbers):
            raise ValueError("a native stack needs >= 2 distinct PRs")

        def snapshot() -> dict[str, Any] | None:
            stack = self.stack_of(numbers[0])
            if stack is None:
                return None
            if stack["pull_requests"][:len(numbers)] != numbers:
                raise PublishError(f"PR #{numbers[0]} already in stack #{stack['number']} with other layers")
            return stack

        def revalidate() -> None:
            for n in numbers[1:]:
                if self.stack_of(n) is not None:
                    raise PublishError(f"PR #{n} already belongs to a stack")

        def mutate() -> dict[str, Any]:
            args = ["gh", "api", "-X", "POST", f"repos/{self.slug}/stacks"]
            for n in numbers:
                args += ["-F", f"pull_requests[]={n}"]
            out = self._mutate(args)
            if self.dry_run:
                return {"number": _synthetic_number(self.slug, "stack", numbers[0]), "pull_requests": numbers}
            return _parse_stack(json.loads(out)) or {}

        def verify(_r: Any) -> bool:
            stack = self.stack_of(numbers[-1])
            return bool(stack and stack["pull_requests"][:len(numbers)] == numbers)

        return self._guarded(self._op("stack-create", *numbers), snapshot=snapshot, revalidate=revalidate,
                             mutate=mutate, verify=verify)

    def add_to_stack(self, stack_number: int, pr: int, *, below: int) -> dict[str, Any]:
        s, n, prev = validate_pr_number(stack_number), validate_pr_number(pr), validate_pr_number(below)

        def snapshot() -> dict[str, Any] | None:
            stack = self.stack_of(n)
            if stack is None:
                return None
            if stack["number"] != s:
                raise PublishError(f"PR #{n} belongs to stack #{stack['number']}, expected #{s}")
            return stack

        def revalidate() -> None:
            if self.dry_run:
                return
            stack = self.stack_of(prev)
            if stack is None or stack["number"] != s or stack["pull_requests"][-1:] != [prev]:
                raise PublishError(f"stack #{s} top is not PR #{prev}")

        def mutate() -> dict[str, Any]:
            out = self._mutate(["gh", "api", "-X", "POST", f"repos/{self.slug}/stacks/{s}/add",
                                "-F", f"pull_requests[]={n}"])
            if self.dry_run:
                return {"number": s, "pull_requests": [prev, n]}
            return _parse_stack(json.loads(out)) if out.strip() else {"number": s}

        def verify(_r: Any) -> bool:
            stack = self.stack_of(n)
            return bool(stack and stack["number"] == s)

        return self._guarded(self._op("stack-add", s, n), snapshot=snapshot, revalidate=revalidate,
                             mutate=mutate, verify=verify)

    # ---- round / land
    def publish_round(self, *, eid: str, winner: str | None, heads: Mapping[str, str],
                      stack: Mapping[str, Any], title: str, body: str) -> dict[str, Any]:
        state = copy.deepcopy(dict(stack)) or {"layers": [], "stack_number": None}
        layers: list[dict[str, Any]] = state.setdefault("layers", [])
        heads = {arm: validate_sha(sha) for arm, sha in heads.items()}
        out: dict[str, Any] = {"winner": winner, "pr": None, "archived": []}
        if winner is not None:
            branch = arm_branch(eid, winner)
            sha = heads[winner]
            idx = next((i for i, layer in enumerate(layers) if layer["eid"] == eid), len(layers))
            base = layers[idx - 1]["branch"] if idx > 0 else self.base
            self.push_branch(branch, sha)
            pr = self.open_pr(branch, base, title, body, self.draft)
            if idx == len(layers):
                layers.append({"eid": eid, "arm": winner, "branch": branch, "head": sha, "pr": pr["number"]})
            out["pr"] = pr["number"]
            if len(layers) >= 2:
                if state.get("stack_number") is None:
                    created = self.create_stack([layer["pr"] for layer in layers])
                    state["stack_number"] = created["number"]
                else:
                    self.add_to_stack(state["stack_number"], layers[-1]["pr"], below=layers[-2]["pr"])
        for arm, sha in sorted(heads.items()):
            if arm != winner:
                out["archived"].append(self.archive_tag(eid, arm, sha)["ref"])
        out["stack"] = state
        return out

    def land(self, stack: Mapping[str, Any]) -> dict[str, Any]:
        """Mark every layer ready, then merge the top PR (lands the native stack)."""
        layers = list(stack.get("layers") or [])
        if not layers:
            raise PublishError("nothing to land")
        for layer in layers:
            n = validate_pr_number(layer["pr"])

            def ready_snapshot(n: int = n) -> dict[str, Any] | None:
                view = self.pr_view(n)
                return {"number": n, "ready": True} if view and not view.get("isDraft") else None

            self._guarded(self._op("ready", n), snapshot=ready_snapshot, revalidate=lambda: None,
                          mutate=lambda n=n: (self._mutate(["gh", "pr", "ready", str(n), "--repo", self.slug]),
                                              {"number": n, "ready": True})[1],
                          verify=lambda _r, n=n: ready_snapshot(n) is not None)
        top = validate_pr_number(layers[-1]["pr"])

        def merge_snapshot() -> dict[str, Any] | None:
            view = self.pr_view(top)
            return {"number": top, "state": "MERGED"} if view and view.get("state") == "MERGED" else None

        merged = self._guarded(
            self._op("merge", top), snapshot=merge_snapshot, revalidate=lambda: None,
            mutate=lambda: (self._mutate(["gh", "pr", "merge", str(top), "--repo", self.slug, "--merge", "--auto"]),
                            {"number": top, "state": "AUTO_MERGE"})[1],
            verify=lambda _r: True)
        return {"top": top, "merge": merged, "layers": [layer["pr"] for layer in layers]}
