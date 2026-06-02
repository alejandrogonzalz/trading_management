# Contributing

## Setup

```bash
pip install pre-commit
pre-commit install          # installs git hooks
pre-commit install --hook-type commit-msg  # installs commit-msg hook
```

## Commit format

This repo enforces [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): short description

# Types
feat      New feature
fix       Bug fix
refactor  Code change that neither fixes a bug nor adds a feature
docs      Documentation only
test      Adding or fixing tests
chore     Tooling, deps, CI
perf      Performance improvement
ci        CI/CD changes

# Scopes (optional but recommended)
langgraph   backend   frontend   optimization   backtest   docs
```

### Examples

```
feat(langgraph): add QLoRA evaluation pipeline with trade simulation
fix(backend): correct fee-aware clipping for BNB pairs
docs(optimization): update SageMaker guide for current train_qlora pipeline
test(backtest): add McNemar and paired t-test unit tests
chore: pin backend requirements and add pre-commit hooks
```

## Running tests

```bash
# LangGraph
cd langgraph && python -m pytest tests/backtest/ tests/optimization/ -v

# Backend
cd backend && python -m pytest tests/ -v
```

## Linting

```bash
ruff check langgraph/ backend/    # lint
ruff format langgraph/ backend/   # format
```
