"""``python -m ci_lab.chat.launch <draft.json>``: run a drafted campaign's local chain.

Spawned detached by the chat server's ``launch_campaign`` (stdout/stderr go to
``<chat_dir>/launches/<cid>.log``). Runs ``campaign new -> calibrate -> run`` one after the
other, stops at the first failure and records progress in
``<chat_dir>/launches/<cid>.status.json`` (``status`` is running, succeeded or failed).
Only ``python -m ci_lab.cli campaign ...`` argv lists from the draft are accepted; nothing
goes through a shell.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from ci_lab.contracts import CAMPAIGN_RE

STEPS = ("new", "calibrate", "run")


def _write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def validate_commands(draft: dict[str, Any]) -> list[list[str]]:
    cid = draft.get("cid")
    if not isinstance(cid, str) or not CAMPAIGN_RE.match(cid):
        raise ValueError(f"bad campaign id {cid!r}")
    if draft.get("target") != "local":
        raise ValueError("only local drafts are launched by this module")
    commands = draft.get("commands")
    if not isinstance(commands, list) or len(commands) != len(STEPS):
        raise ValueError("draft has no new/calibrate/run commands")
    for step, argv in zip(STEPS, commands, strict=True):
        if not (isinstance(argv, list) and all(isinstance(a, str) for a in argv)
                and argv[1:6] == ["-m", "ci_lab.cli", "campaign", step, cid]):
            raise ValueError(f"unexpected {step} command: {argv!r}")
    return commands


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m ci_lab.chat.launch <draft.json>", file=sys.stderr)
        return 2
    draft_path = Path(args[0]).resolve()
    draft = json.loads(draft_path.read_text(encoding="utf-8"))
    commands = validate_commands(draft)
    status_path = draft_path.parent.parent / "launches" / f"{draft['cid']}.status.json"
    status: dict[str, Any] = {"cid": draft["cid"], "status": "running", "step": None, "pid": os.getpid(),
                              "started": time.time()}
    for step, cmd in zip(STEPS, commands, strict=True):
        status["step"] = step
        _write(status_path, status)
        print(f"== ci-lab chat launch: {step}: {cmd}", flush=True)
        rc = subprocess.call(cmd, stdin=subprocess.DEVNULL)
        if rc != 0:
            status.update(status="failed", returncode=rc, finished=time.time())
            _write(status_path, status)
            print(f"== {step} failed with exit code {rc}", flush=True)
            return rc
    status.update(status="succeeded", returncode=0, finished=time.time())
    _write(status_path, status)
    print("== campaign chain finished", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
