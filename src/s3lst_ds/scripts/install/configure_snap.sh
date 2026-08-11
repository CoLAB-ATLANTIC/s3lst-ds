#!/usr/bin/env bash

# Configure snappy

# ---> Get absolute path to scripts directory
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---> Get utility functions
source "$SCRIPTS_DIR/utils.sh"

# Get the path to the python binary of the activated virtual environment
PYTHON_BIN="$(python -c 'import sys; print(sys.executable)')"

# Associates the activated virtual environment (which contains snappy) with the SNAP
# installation
info "Configuring snappy..."
if "$HOME/esa-snap/bin/snappy-conf" "$PYTHON_BIN"; then
    info "Done."
else
    error "Failed."
    return 1
fi