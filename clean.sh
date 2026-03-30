#!/usr/bin/env bash
# Remove Cython build artifacts only. Does NOT touch .venv or Python source files.

# Compiled Cython extensions and generated C files
find . -path './.venv' -prune -o -name "*.so" -exec rm -v {} \;
find . -path './.venv' -prune -o -name "*.c" -path "*/dynamics/accretion/*" -exec rm -v {} \;
find . -path './.venv' -prune -o -name "*.c" -path "*/math_utils/*" -exec rm -v {} \;
find . -path './.venv' -prune -o -name "*.c" -path "*/radiation/opacity/*" -exec rm -v {} \;

# setuptools / pip build artifacts
rm -rvf build dist triceratops.egg-info
