"""Lessons-as-structure rule engine (design §13, §13.6, §13.7).

Rules are data (``ci_lab.rulespec``); this package is the frozen interpreter the model cannot
bypass. Pure: no network, no LLM, no OTel global state; only the loaders touch the filesystem.
"""

