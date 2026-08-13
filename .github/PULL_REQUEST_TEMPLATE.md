## What does this change?

<!-- One or two sentences. -->

## Why?

<!-- What problem does it solve? Link an issue if there is one. -->

## Validation

If this touches signal generation, backtesting or validation, state the evidence:

- [ ] `pytest` passes
- [ ] `ruff check .` and `mypy` pass
- [ ] The null test still reports no edge: `cryptoquant selftest --trend 0.0`
- [ ] For strategy changes: deflated Sharpe, PBO and walk-forward numbers are in the description
- [ ] Number of parameter variants tried is stated honestly

## Risk

- [ ] This does not weaken any risk limit, kill switch, or the `testnet` / `dry_run` defaults
