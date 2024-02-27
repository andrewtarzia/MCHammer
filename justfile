# List all recipes.
default:
  @just --list

# Install development environment.
dev:
  pip install -e '.[dev]'

# Run code checks.
check:
  #!/usr/bin/env bash

  error=0
  trap error=1 ERR

  echo
  (set -x; ruff . )

  echo
  ( set -x; ruff format --check . )

  echo
  ( set -x; mypy mchammer )

  echo
  ( set -x; pytest --cov=mchammer --cov-report term-missing )

  test $error = 0


# Auto-fix code issues.
fix:
  ruff format .
  ruff --fix .


# Build docs.
docs:
  rm -rf docs/source/_autosummary
  make -C docs html
  echo Docs are in $PWD/docs/build/html/index.html
