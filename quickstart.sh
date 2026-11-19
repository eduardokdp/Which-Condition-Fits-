#!/usr/bin/env bash
# Shortcut: runs proposed-solution/quickstart.sh, whatever the current folder.
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/proposed-solution/quickstart.sh" "$@"
