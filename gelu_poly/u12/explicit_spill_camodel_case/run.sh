#!/usr/bin/env bash
set -e
case_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${case_dir}/set_env.sh"
exec python3 "${case_dir}/run.py" "$@"
