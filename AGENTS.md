# Repository guidance

## Scope

This repository analyzes Nikkei 225 seasonality and market valuation with Python. Keep historical valuation point-in-time: do not use observations that were unavailable on the evaluated date.

## Tooling

- Python: 3.10+
- Dependency management: `uv`
- Lint/format: Ruff
- Tests: pytest
- Task runner: Task

## Commands

```bash
uv sync
uv run pytest
uv run ruff check .
task format
task seasonality YEARS=5
task valuation-ts YEARS=5 PREMIUM=3.5
uv run python main.py valuation --current-per 19.75
```

`Taskfile.yml`, `pyproject.toml`, and `main.py` are the authority for available commands. Do not document a task that is not defined there.

## Structure

- `main.py`: CLI entry point
- `src/`: analysis, data, risk, options, and visualization code
- `tests/`: regression tests
- `scripts/`: report generation
- `docs/`: generated/public Pages content
- `data/`: analysis inputs and outputs tracked by the repository

## Changes

- Prefer deleting or consolidating unused code, configuration, and documentation over adding alternatives.
- Do not duplicate calculation logic between CLI, reports, and browser-facing code.
- Keep public behavior, tests, and the Pages deployment path working when simplifying.
- Use plain language and established Python, GitHub Actions, and financial terminology.
- Run only the checks relevant to the change and report unexecuted checks as unverified.
