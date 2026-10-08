"""Programmatic optimizers for arm strategies (design §11): DSPy LM routing, GEPA and
SkillOpt adapters over harness text components.

Importing this package never imports ``dspy``/``gepa``/``skillopt_sleep`` (C26); the
heavy libraries load inside the functions that need them.
"""
