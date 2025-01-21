#!/bin/sh -e
set -x

autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place "src" "run.py" --exclude=__init__.py
isort "src" "run.py" 
black "src" "run.py" -l 80
