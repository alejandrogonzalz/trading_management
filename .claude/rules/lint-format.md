# Lint & Format — Ruff

The project's CI runs **two separate Ruff commands**. Passing one does NOT imply
passing the other — both must be green before committing Python changes.

| Command | Tool | Catches |
|---------|------|---------|
| `ruff check langgraph/ backend/ --exclude="*.ipynb"` | Linter | unused imports, unsorted imports (I001), undefined names, unused variables (F841) |
| `ruff format --check langgraph/ backend/ --exclude="*.ipynb"` | Formatter (dry-run) | style drift: comment spacing, slice spacing (`x[a:b]` → `x[a :b]`), quote style, line wrapping |

`--check` means "report only, don't write" → it **exits non-zero whenever a file
is not byte-identical to what the formatter would produce.** That is the usual
cause of a surprise "lint" failure: the linter passes but the formatter check
fails because a file was hand-edited and never run through `ruff format`.

## Before committing any Python change, run BOTH and apply fixes

```bash
# From repo root (trading_management/)
ruff check  langgraph/ backend/ --exclude="*.ipynb" --fix   # linter, auto-fixes imports etc.
ruff format langgraph/ backend/ --exclude="*.ipynb"          # formatter, rewrites style

# Then verify both are clean (this is what CI runs):
ruff check        langgraph/ backend/ --exclude="*.ipynb"
ruff format --check langgraph/ backend/ --exclude="*.ipynb"
```

## Rules

1. **Never hand-format Python to "match" the linter.** Run `ruff format` and let
   it own whitespace/line-wrap/quote decisions. Manual spacing will fail
   `ruff format --check` even when `ruff check` is green.
2. **Run `ruff format` (not just `ruff check`) after editing any `.py` file** in
   `langgraph/` or `backend/`, including test files. Formatter-only style changes
   in 8 files once slipped through because only the linter was run.
3. **Keep formatting commits isolated.** When a bulk `ruff format` touches many
   files, commit it on its own as `style: apply ruff format` so logic diffs stay
   reviewable. Don't mix a large cosmetic reformat into a feature/fix commit.
4. Notebooks are excluded (`--exclude="*.ipynb"`) — do not run ruff on `.ipynb`.
5. The two commands are different tools. If CI says "would reformat", that's the
   **formatter**, fix it with `ruff format`. If CI names a rule code (I001, F401,
   F841…), that's the **linter**, fix it with `ruff check --fix` or by hand.
