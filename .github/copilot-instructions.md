# stable-retro-scripts Copilot Instructions

- Assume command examples are run from the repository root. Prefer paths like `python3 scripts/train.py` and `python3 scripts/play.py` in docs and generated commands.
- Keep the current direct-import script layout working. Files under `scripts/` are executed as scripts, so avoid refactoring them into package-relative imports unless the task is explicitly a packaging restructure.
- Keep Python RL/training logic under `scripts/` and C++ runtime/inference code under `retro_ai_lib/`. Do not mix concerns unless the task requires coordinated changes.
- Treat `models/`, `screenshots/`, and `retro_ai_lib/data/` as data or binary assets. Do not rewrite, re-export, or regenerate them unless the task explicitly asks for that.
- `stable-retro` is a sibling dependency used by this repo. Do not modify the separate `stable-retro` repo unless the task explicitly includes it.
- For C++ changes, follow `.clang-format` and the `retro_ai_lib/` CMake workflow. Do not reference the legacy `ef_lib` path in docs, CI, or commands.
- Preserve existing CLI flags, reward-function names, and state names because curricula, readmes, and saved model workflows depend on them.
- When validating changes, prefer the repo's existing workflows: `pylint --rcfile=.pylintrc $(git ls-files '*.py')` for Python and the `retro_ai_lib` CMake build for C++ when libtorch/OpenCV are available.