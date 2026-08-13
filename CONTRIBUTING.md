# Contributing

Thanks for considering a contribution.

## Setup

```bash
git clone https://github.com/flavianosantos/cryptoquant
cd cryptoquant
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev,live]"
pre-commit install
```

## Before opening a pull request

```bash
ruff check . && ruff format --check .
mypy
pytest
cryptoquant selftest --trend 0.0        # the process must find no edge in noise
```

`pre-commit install` runs the first two automatically on every commit.

## Code conventions

- **Docstrings are required** on every public module, class and function, in
  Google style. Explain *why*, not just *what* — the reasoning behind a
  statistical choice is the part a reader cannot reconstruct from the code.
- **Type hints on all public signatures.** `mypy` runs in CI.
- **No lookahead, ever.** Any function producing a feature or signal must use
  only information available at or before the bar it is stamped on. If you are
  not certain, add a case to `tests/test_no_lookahead.py`.
- **Costs are not optional.** New execution paths charge fees and slippage.
- Keep dependencies minimal. Anything new needs a justification in the PR.

## Contributing a strategy

A pull request adding a strategy is judged on its evidence, not its returns:

1. **Economic rationale.** Why should this work? "The backtest looks good" is
   not a rationale — it is the thing that needs explaining.
2. **Honest trial count.** State how many variants you tried, including
   parameters you adjusted by hand. All of them count.
3. **Validation numbers in the PR description**: deflated Sharpe, PBO, and a
   walk-forward equity curve. See the gates in the README.
4. **Beat both benchmarks** — buy-and-hold and `MovingAverageCrossover` — net
   of costs, or explain clearly why it is interesting despite not doing so.

A strategy with a Sharpe of 0.7 and a deflated Sharpe of 0.95 is a better
contribution than one with a Sharpe of 2.5 and a PBO of 0.8.

## Reporting security issues

Do not open a public issue for anything involving credentials or key handling.
Email the maintainer instead.
