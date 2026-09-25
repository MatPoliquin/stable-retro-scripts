# stable-retro-scripts Copilot Instructions

- Assume command examples are run from the repository root. Prefer paths like `python3 scripts/train.py` and `python3 scripts/play.py` in docs and generated commands.
- Keep the current direct-import script layout working. Files under `scripts/` are executed as scripts, so avoid refactoring them into package-relative imports unless the task is explicitly a packaging restructure.
- Keep Python RL/training and model export logic under `scripts/`. C++ runtime/inference code now lives in the separate retro-ai-runtime repository; do not modify that sibling unless the task includes it.
- Treat `models/` and `screenshots/` as data or binary assets. Do not rewrite, re-export, or regenerate them unless the task explicitly asks for that.
- `stable-retro` is a sibling dependency used by this repo. Do not modify the separate `stable-retro` repo unless the task explicitly includes it.
- Preserve existing CLI flags, reward-function names, and state names because curricula, readmes, and saved model workflows depend on them.
- Use Python 3.10 through 3.12. Install runtime dependencies from `requirements.txt`, contributor dependencies from `requirements-dev.txt`, and lint-only CI dependencies from `requirements-lint.txt`; do not replace these with ad hoc package installs.
- When validating changes, prefer the repo's existing Python workflow: `pylint --rcfile=.pylintrc $(git ls-files '*.py')`, plus relevant tests under `tests/`. C++ build and test workflows now live in retro-ai-runtime.
