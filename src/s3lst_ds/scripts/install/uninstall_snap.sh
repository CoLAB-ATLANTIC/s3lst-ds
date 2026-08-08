#!/usr/bin/env bash

# Uninstall SNAP dependencies

# ---> Get absolute path to scripts directory
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---> Get utility functions
source "$SCRIPTS_DIR/utils.sh"

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
