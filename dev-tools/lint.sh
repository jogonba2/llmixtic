#!/usr/bin/env bash

set -e
set -x

mypy "src" "run.py"
flake8 "src" "run.py" --ignore=E501,W503,E203,E402
black "src" "run.py" --check -l 80
