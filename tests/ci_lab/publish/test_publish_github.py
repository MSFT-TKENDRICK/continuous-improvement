from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import pytest

from ci_lab.publish.github import (
    CommandError,
    GitHubPublisher,
    PublishError,
    archive_tag_name,
    validate_pr_number,
    validate_ref,
    validate_repo,
    validate_sha,
)
from ci_lab.testing import MemoryOutbox

SHA1 = "a" * 40
SHA2 = "b" * 40
SHA3 = "c" * 40


class FakeGitHub:
    """Tiny in-memory model of the remote + gh REST surface used by the publisher."""

    def __init__(self) -> None:
        self.refs: dict[str, str] = {"refs/heads/main": "f" * 40}
        self.prs: dict[int, dict[str, Any]] = {}
        self.stacks: dict[int, list[int]] = {}
        self.argv: list[list[str]] = []
        self.fail_next: str | None = None

    def __call__(self, args: Sequence[str]) -> str:
        a = list(args)
        self.argv.append(a)
        if self.fail_next and self.fail_next in " ".join(a):
            self.fail_next = None
            raise CommandError(a, 1, "transient")
        if a[:2] == ["git", "ls-remote"]:
            ref = a[3]
            return f"{self.refs[ref]}\t{ref}\n" if ref in self.refs else ""
        if a[:2] == ["git", "push"]:
            sha, ref = a[3].split(":")
            self.refs[ref] = sha
            return ""
        if a[:3] == ["gh", "pr", "list"]:
            head = a[a.index("--head") + 1]
            return json.dumps([{"number": n, "headRefName": p["head"], "baseRefName": p["base"],
                                "state": p["state"], "isDraft": p["draft"], "url": f"u/{n}"}
                               for n, p in self.prs.items() if p["head"] == head])
        if a[:3] == ["gh", "pr", "create"]:
            n = 100 + len(self.prs) + 1
            self.prs[n] = {"head": a[a.index("--head") + 1], "base": a[a.index("--base") + 1],
                           "state": "OPEN", "draft": "--draft" in a}
            return f"https://github.com/o/r/pull/{n}\n"
        if a[:3] == ["gh", "pr", "view"]:
            p = self.prs[int(a[3])]
            return json.dumps({"number": int(a[3]), "state": p["state"], "isDraft": p["draft"]})
        if a[:3] == ["gh", "pr", "ready"]:
            self.prs[int(a[3])]["draft"] = False
            return ""
        if a[:3] == ["gh", "pr", "merge"]:
            for prs in self.stacks.values():
                if int(a[3]) in prs:
                    for n in prs:
                        self.prs[n]["state"] = "MERGED"
            return ""
        if a[:2] == ["gh", "api"]:
            prs = [int(x.split("=")[1]) for x in a if x.startswith("pull_requests[]=")]
            if a[2] != "-X":
                n = int(a[2].split("pull_request=")[1])
                for s, members in self.stacks.items():
                    if n in members:
                        return json.dumps({"number": s, "pull_requests": [{"number": m} for m in members]})
                raise CommandError(a, 1, "gh: Not Found (HTTP 404)")
            path = a[4]
            if path.endswith("/stacks"):
                s = 7 + len(self.stacks)
                self.stacks[s] = prs
                return json.dumps({"number": s, "pull_requests": [{"number": m} for m in prs]})
            s = int(path.split("/")[-2])
            self.stacks[s] += prs
            return json.dumps({"number": s, "pull_requests": [{"number": m} for m in self.stacks[s]]})
        raise AssertionError(f"unexpected command {a}")

    def mutations(self) -> list[list[str]]:
        return [a for a in self.argv if a[:2] == ["git", "push"] or a[:3] in (
            ["gh", "pr", "create"], ["gh", "pr", "ready"], ["gh", "pr", "merge"]) or (a[:2] == ["gh", "api"]
                                                                                    and a[2] == "-X")]


def _pub(gh: FakeGitHub | None = None, outbox: MemoryOutbox | None = None, **kw: Any) -> GitHubPublisher:
    return GitHubPublisher("octo/harness", outbox=outbox or MemoryOutbox(), runner=gh or FakeGitHub(), **kw)


# ------------------------------------------------------------------ validation

@pytest.mark.parametrize("value", ["1;rm -rf /", "-1", "0", "12a", "", " 1", "1\n", "1" * 11, True, 1.5, None])
def test_rejects_bad_pr_numbers(value: Any) -> None:
    with pytest.raises(ValueError):
        validate_pr_number(value)


@pytest.mark.parametrize("value", ["A" * 40, "a" * 39, "a" * 41, "g" * 40, "a" * 40 + "\n", "HEAD", "-" + "a" * 39])
def test_rejects_bad_shas(value: str) -> None:
    with pytest.raises(ValueError):
        validate_sha(value)


@pytest.mark.parametrize("value", [
    "--upload-pack=touch x", "-b", "a..b", "a b", "a~1", "a^", "a:b", "x.lock", ".hidden", "a//b", "a/",
    "$(id)", "a;b", "a|b", "HEAD", "a\\b", "refs/heads/../x", "a/.b", "", "a@{1}", "\u00e9", "a\nb"])
def test_rejects_bad_refs(value: str) -> None:
    with pytest.raises(ValueError):
        validate_ref(value)


@pytest.mark.parametrize("value", ["octo/harness/x", "../harness", "octo/-r", "octo;/r", "octo/..", "octo",
                                   "octo/r r", "octo/$(id)"])
def test_rejects_bad_repos(value: str) -> None:
    with pytest.raises(ValueError):
        validate_repo(value)


def test_accepts_good_values() -> None:
    assert validate_pr_number("42") == 42
    assert validate_sha("0" * 64) == "0" * 64
    assert validate_ref("exp/demo-camp-r01/v1") == "exp/demo-camp-r01/v1"
    assert validate_repo("octo-org/harness.v2") == ("octo-org", "harness.v2")
    assert archive_tag_name("demo-camp-r01", "v2") == "exp-archive/demo-camp-r01/v2"
    with pytest.raises(ValueError):
        archive_tag_name("demo-camp-r01", "../x")


def test_malicious_inputs_never_reach_the_runner() -> None:
    gh = FakeGitHub()
    pub = _pub(gh)
    with pytest.raises(ValueError):
        pub.find_pr("--help")
    with pytest.raises(ValueError):
        pub.open_pr("exp/a-b-r01/v1", "main;rm", "t", "b")
    with pytest.raises(ValueError):
        pub.open_pr("exp/a-b-r01/v1", "main", "bad\x1btitle", "b")
    with pytest.raises(ValueError):
        pub.stack_of("12 --method DELETE")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        pub.create_stack([101])
    with pytest.raises(ValueError):
        pub.publish_round(eid="demo-camp-r01", winner="v1", heads={"v1": "HEAD~1"}, stack={}, title="t", body="b")
    with pytest.raises(ValueError):
        pub.publish_round(eid="demo-camp-r01", winner="../x", heads={"../x": SHA1}, stack={}, title="t", body="b")
    with pytest.raises(ValueError):
        GitHubPublisher("octo/harness", outbox=MemoryOutbox(), runner=gh, remote="--exec=x")
    assert gh.argv == []


# ------------------------------------------------------------------ live (fake gh) behaviour

def _rounds(pub: GitHubPublisher) -> dict[str, Any]:
    stack: dict[str, Any] = {"layers": [], "stack_number": None}
    for r, (win, lose) in enumerate([(SHA1, SHA2), (SHA2, SHA3), (SHA3, None)], start=1):
        heads = {"v1": win} | ({"v2": lose} if lose else {})
        out = pub.publish_round(eid=f"demo-camp-r0{r}", winner="v1", heads=heads, stack=stack,
                                title=f"round {r}", body="body")
        stack = out["stack"]
    return stack


def test_publish_rounds_build_native_stack() -> None:
    gh = FakeGitHub()
    pub = _pub(gh)
    stack = _rounds(pub)
    assert [layer["pr"] for layer in stack["layers"]] == [101, 102, 103]
    assert gh.prs[101]["base"] == "main"
    assert gh.prs[102]["base"] == "exp/demo-camp-r01/v1"
    assert gh.prs[103]["base"] == "exp/demo-camp-r02/v1"
    assert gh.stacks == {stack["stack_number"]: [101, 102, 103]}
    assert gh.refs["refs/tags/exp-archive/demo-camp-r01/v2"] == SHA2
    assert not any(p["head"].endswith("/v2") for p in gh.prs.values())  # losers get tags, not PRs
    create = next(a for a in gh.argv if a[:3] == ["gh", "api", "-X"] and a[4].endswith("/stacks"))
    assert create[5:] == ["-F", "pull_requests[]=101", "-F", "pull_requests[]=102"]
    add = next(a for a in gh.argv if a[:3] == ["gh", "api", "-X"] and a[4].endswith("/add"))
    assert add[4] == f"repos/octo/harness/stacks/{stack['stack_number']}/add" and add[5:] == ["-F",
                                                                                            "pull_requests[]=103"]

    landed = pub.land(stack)
    assert landed["top"] == 103
    assert all(not p["draft"] for p in gh.prs.values()) and gh.prs[101]["state"] == "MERGED"


def test_rerun_with_same_outbox_is_noop() -> None:
    gh = FakeGitHub()
    outbox = MemoryOutbox()
    _rounds(_pub(gh, outbox))
    before = len(gh.mutations())
    _rounds(_pub(gh, outbox))
    assert len(gh.mutations()) == before


def test_fresh_outbox_reconciles_from_remote_state() -> None:
    gh = FakeGitHub()
    _rounds(_pub(gh))
    before = len(gh.mutations())
    _rounds(_pub(gh, MemoryOutbox()))  # lost journal: snapshots find branch/PR/stack/tag
    assert len(gh.mutations()) == before
    assert len(gh.prs) == 3


def test_crash_mid_round_resumes_without_duplicates() -> None:
    gh = FakeGitHub()
    outbox = MemoryOutbox()
    gh.fail_next = "pr create"
    with pytest.raises(CommandError):
        _rounds(_pub(gh, outbox))
    _rounds(_pub(gh, outbox))
    pushes = [a for a in gh.mutations() if a[:2] == ["git", "push"]]
    assert len(pushes) == len({tuple(a) for a in pushes}) == 5
    assert len(gh.prs) == 3


def test_revalidate_and_snapshot_refuse_conflicts() -> None:
    gh = FakeGitHub()
    pub = _pub(gh)
    gh.refs["refs/heads/exp/demo-camp-r01/v1"] = SHA3
    with pytest.raises(PublishError, match="another commit"):
        pub.push_branch("exp/demo-camp-r01/v1", SHA1)
    with pytest.raises(PublishError, match="not on"):
        pub.open_pr("exp/demo-camp-r09/v1", "main", "t", "b")
    gh.prs[555] = {"head": "exp/demo-camp-r01/v1", "base": "main", "state": "CLOSED", "draft": True}
    with pytest.raises(PublishError, match="CLOSED"):
        pub.open_pr("exp/demo-camp-r01/v1", "main", "t", "b")


# ------------------------------------------------------------------ dry run

def test_dry_run_records_calls_and_never_runs(tmp_path) -> None:
    def runner(args: Sequence[str]) -> str:
        raise AssertionError(f"dry run executed {args}")

    journal = tmp_path / "calls.jsonl"
    pub = GitHubPublisher("octo/harness", outbox=MemoryOutbox(), runner=runner, dry_run=True, journal=journal)
    stack = _rounds(pub)
    assert [c[:3] for c in pub.calls] == [
        ["git", "push", "origin"], ["gh", "pr", "create"], ["git", "push", "origin"],
        ["git", "push", "origin"], ["gh", "pr", "create"], ["gh", "api", "-X"], ["git", "push", "origin"],
        ["git", "push", "origin"], ["gh", "pr", "create"], ["gh", "api", "-X"]]
    assert len(stack["layers"]) == 3 and stack["stack_number"]
    assert len(journal.read_text().splitlines()) == len(pub.calls)
    assert all(json.loads(line)["dry_run"] for line in journal.read_text().splitlines())
