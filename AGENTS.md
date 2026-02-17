## No silent failures / no ambiguity

This is research code: correctness and clarity beat cleverness.

- **Never fail silently.** No empty `except`, no “best-effort” fallbacks, no ignoring return codes, no `pass` on errors.
- **Make errors loud and informative.** Raise explicit exceptions with actionable messages; include context (inputs, shapes, units, paths, assumptions).
- **Be explicit, not ambiguous.** Prefer readable code over implicit magic; avoid unclear defaults and side effects.
- **Validate inputs + invariants.** Assert/guard preconditions and key assumptions early (types, ranges, dimensions, units).
- **If uncertain, stop and say so.** Don’t guess—surface the uncertainty and propose a safe, checkable approach.

Silent errors are worse than crashes. Crashes with good messages are acceptable.