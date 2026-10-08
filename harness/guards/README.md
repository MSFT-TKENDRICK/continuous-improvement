# Guards

Guard rule files (`*.yaml`, `ci_lab.guards` / `ci_lab.lessons_arm` format) for the target agents live here.
Only the `guard` strategy may write them (`BUNDLE.lock` is never written by a candidate); text strategies
(agent, gepa, skillopt) never touch `guards/**`. The tree ships no rules yet: the guard arm adds them, and
`HarnessTree.validate()` only requires that every file here belongs to the `guard` component.
