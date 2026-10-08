"""Dev-loop lessons as structure (design §13.1 R5, §13.4): ``ci-lab lint`` + ``ci-lab reflect``."""

from ci_lab.lint.engine import Finding, LintResult, format_json, format_text, lint, run
from ci_lab.lint.spec import (
           LintRule,
           LintRuleFile,
           RuleLoadError,
           load_rules,
           parse_rules,
           rule_files,
)

__all__ = ["Finding", "LintResult", "LintRule", "LintRuleFile", "RuleLoadError", "format_json", "format_text",
           "lint", "load_rules", "parse_rules", "rule_files", "run"]
