#!/usr/bin/env bash

# Exit on error (-e), on use of undefined variables (-u), and if any command in a
# pipeline fails (-o pipefail).
# For details, see: https://linuxcommand.org/lc3_man_pages/seth.html
set -euo pipefail

# ---> Get helpful paths
# Get absolute path to this very script
SCRIPT_FILE="$(realpath "${BASH_SOURCE[0]}")"
# Get absolute path to parent directory of this very script (install)
INSTALL_DIR="$(dirname "$SCRIPT_FILE")"
# Get absolute path to parent directory of the install directory (scripts)
SCRIPTS_DIR="$(dirname "$INSTALL_DIR")"

# ---> Get custom printing functions as well as text style variables
source "$SCRIPTS_DIR/utils/print.sh"

# ---> Uninstall SNAP dependencies
info "Removing SNAP main installation directory (esa-snap)..."
if [ -d "$HOME/esa-snap" ]; then
    if rm -rf "$HOME/esa-snap"; then
        info "Done."
    else
        error "Failed."
        return 1
    fi
else
    info "Skipped: directory not found."
fi

info "Removing SNAP config directory (.snap)..."
if [ -d "$HOME/.snap" ]; then
    if rm -rf "$HOME/.snap"; then
        info "Done."
    else
        error "Failed."
        return 1
    fi
else
    info "Skipped: directory not found."
fi

info "Removing SNAP desktop entry (*snap*.desktop)..."
if compgen -G "$HOME/.local/share/applications/*snap*.desktop" > /dev/null; then
    if rm -f $HOME/.local/share/applications/*snap*.desktop; then
        info "Done."
    else
        error "Failed."
        return 1
    fi
else
    info "Skipped: file not found."
fi
