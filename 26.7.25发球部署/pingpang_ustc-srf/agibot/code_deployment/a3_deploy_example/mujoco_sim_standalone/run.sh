#!/usr/bin/env bash
set -euo pipefail

package_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec "$package_dir/bin/start_mujoco_sim.sh" "$@"
