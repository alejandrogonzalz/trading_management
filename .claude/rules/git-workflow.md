# Git Workflow

## Branching
- `main` — production-ready, merged via PR only
- `dev` — integration branch
- `feature/*` — new features (e.g., feature/avance5-ensembles)
- Commit messages follow conventional commits: `feat()`, `fix()`, `refactor()`, `docs()`

## Commit Conventions
- Scope: `langgraph`, `backend`, `frontend`, or omit for root
- Keep commits atomic: one logical change per commit
- Never commit `.env`, `*.db`, model weights (`.pt`, `.pkl`) unless tracked by DVC
- Notebooks must be executed before committing (all cells have outputs)

## PR Pattern
- Branch from `dev`, PR back to `dev`
- Avance notebooks go to `feature/avanceN-*` branches
- Squash merge for clean history on main
