"""Lessons → guard rules as an experiment arm (design §13.3 steps 4–8, §13.6 B1–B4, N3–N6).

Deterministic template synthesizers first (:mod:`.synth`), the ``LessonSynthesizer`` MAF agent
only for leftovers (:mod:`.agent`), the ``guard`` :class:`~ci_lab.contracts.ArmStrategy`
(:mod:`.strategy`), paired guard-off/guard-on evaluation (:mod:`.paired`), the OES guard
extension (:mod:`.envelope`), shadow→enforce promotion (:mod:`.promote`), prose deletion
(:mod:`.prose`) and exposure-based retirement (:mod:`.retire`).

Submodules are imported by callers; this package has no import-time side effects.
"""
