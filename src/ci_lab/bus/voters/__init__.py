"""Bus voters (bus contract v2 §9): independent judges that turn one proposal artifact into
``VoteBody`` entries, one per (voter, criterion) they cover.

``local`` holds the in-process / deterministic voters and :func:`~ci_lab.bus.voters.local.run_voters`;
``remote`` holds the ASSERT, System-1 and agent (LLM critic) voters.
"""

__all__: list[str] = []
