"""Arm strategy registry (design §11.2): ``get_strategy(directive.strategy, **deps)``.

Strategies: ``agent`` (MAF meta-agent proposer), ``gepa`` (DSPy GEPA over text components,
``ci_lab.optim``), ``skillopt`` (SkillOpt-Sleep on skill markdown) and ``guard`` (v2.4 §13,
``ci_lab.lessons_arm``; registered lazily on first use). Every strategy emits a
``ci.optimizer`` span, honours ``ArmDirective.edit_budget`` and returns plain
:class:`~ci_lab.contracts.Edit` commits that go through the same critic/ASSERT/RRSI/OES
gates. Importing this package does not import dspy/gepa/skillopt_sleep (C26) or the guard
arm.
"""
