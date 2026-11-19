#!/usr/bin/env bash
# Shortcut: runs proposed-solution/scripts/reproduce_m4.sh, whatever the current folder.
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/proposed-solution/scripts/reproduce_m4.sh" "$@"
