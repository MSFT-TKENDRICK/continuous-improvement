"""Git operations: safe git wrapper, ref names, worktree slot pool, ref publication, safe paths."""

from ci_lab.gitops.git import GitError, tree_hash
from ci_lab.gitops.names import archive_tag, arm_branch, check_ref_format, ledger_branch, sleep_branch
from ci_lab.gitops.publish_refs import PushRejected, TagConflict, is_ancestor, push_with_lease, remote_sha, tag_archive
from ci_lab.gitops.safe_path import UnsafePathError, match_globs, safe_join
from ci_lab.gitops.slots import Slot, SlotPool, SlotPoolExhausted, wt_root

__all__ = [
    "GitError", "PushRejected", "Slot", "SlotPool", "SlotPoolExhausted", "TagConflict", "UnsafePathError",
    "archive_tag", "arm_branch", "check_ref_format", "is_ancestor", "ledger_branch", "match_globs",
    "push_with_lease", "remote_sha", "safe_join", "sleep_branch", "tag_archive", "tree_hash", "wt_root",
]
