#!/bin/bash
# Fork gateway launcher — the AppArmor attachment point.
#
# WHY THIS FILE EXISTS, and why it is not `dev-backend.sh`:
# this host restricts unprivileged user namespaces
# (kernel.apparmor_restrict_unprivileged_userns=1, the Ubuntu 23.10+ default), so
# the sandbox needs a per-application AppArmor profile granting `userns`. AppArmor
# matches a profile attachment against the path the KERNEL RESOLVES, not the
# symlink used to invoke it — and `.venv/bin/python` is a symlink to the shared
# interpreter under ~/.kiro/crew-python. So a profile cannot be attached to the
# venv python (the attachment would never match) and must not be attached to the
# real interpreter (that would grant `userns` to every Python on this machine,
# including the one running the LIVE gateway). A real script at a stable path is
# the only attachment point that confines exactly this instance — the same reason
# `service/apparmor.py` attaches to a launcher rather than to an interpreter.
#
# The profile is a SEPARATE name, `kirocrew-fork-userns`, deliberately: writing
# `kirocrew-userns` would re-point the profile the live install depends on and
# strip its sandbox.
#
# IMPORTANT — invoke the VENV python, never the resolved interpreter. Python
# activates a venv by finding `pyvenv.cfg` relative to the executable path it was
# started with, so calling ~/.kiro/crew-python/.../python3.12 directly bypasses
# the venv entirely and dies on the first dependency (ModuleNotFoundError: yaml).
# The symlink is correct HERE; only the AppArmor attachment needed a real path.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"
if [ ! -x "$VENV_PYTHON" ]; then
    echo "ERROR: $VENV_PYTHON is missing. Run 'make backend' first." >&2
    exit 1
fi

# Same isolation dev-backend.sh sets, defaulted here so the launcher is usable on
# its own: a data home separate from ~/.kiro/crew, and a port separate from the
# live gateway's 5476. Absolutized because config_dir() resolves it against each
# process's CWD, and MCP subprocesses are spawned with session-workspace CWDs — a
# relative HOME makes them create empty config dirs with no .local_secret, so
# their gateway IPC calls fail with 403.
export PYTHONPATH="${PYTHONPATH:-$SCRIPT_DIR/src}"
export KIROCREW_HOME="${KIROCREW_HOME:-$SCRIPT_DIR/.kirocrew-dev}"
case "$KIROCREW_HOME" in
    /*) ;;
    *) KIROCREW_HOME="$SCRIPT_DIR/$KIROCREW_HOME" ;;
esac
export KIROCREW_HOME
export KIROCREW_PORT="${KIROCREW_PORT:-6777}"
export KIROCREW_PROJECT_DIR="${KIROCREW_PROJECT_DIR:-$SCRIPT_DIR}"

echo "👻 Fork gateway (AppArmor-attached launcher, port $KIROCREW_PORT)"
echo "   Python: $VENV_PYTHON"
echo "   Data:   $KIROCREW_HOME"
echo ""

exec "$VENV_PYTHON" -m kiro_crew gateway "$@"
