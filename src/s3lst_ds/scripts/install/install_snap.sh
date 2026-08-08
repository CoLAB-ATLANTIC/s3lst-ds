#!/usr/bin/env bash

# Exit on error (-e), on use of undefined variables (-u), and if any command in a
# pipeline fails (-o pipefail).
# For details, see: https://linuxcommand.org/lc3_man_pages/seth.html
set -euo pipefail

# ---> Get helpful paths
WORKING_DIR="$PWD"
# Get absolute path to scripts directory
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Get absolute path to SNAP installation directory
SNAP_DIR="$HOME/esa-snap"

# ---> Uninstall SNAP dependencies if they were previously installed
source "$SCRIPTS_DIR/install/uninstall_snap.sh"

# ---> Download and install SNAP and configure snappy to use the former
# Change to install directory to download SNAP to it
cd "$SCRIPTS_DIR/install"

# Remove any existing SNAP installer in the install directory to avoid issues with
# multiple file names
if compgen -G "esa-snap_all*" > /dev/null; then
    info "Removing old SNAP installer files..."
    if rm esa-snap_all*; then
        :
    else
        error "Failed."
        cd "$WORKING_DIR"
        return 1
    fi
    info "Done."
fi

# Get the minor of the python version (e.g. 13 in 3.13.12) of the activated virtual
# environment
PYTHON_VERSION_MINOR="$(python -c 'import sys; print(sys.version_info.minor)')"

# Set the SNAP version to install based on the minor of the python version of the
# virtual environment
# NOTE: the latest python version supported by the latest SNAP version usually has as
# minor the major of the latter. E.g SNAP 13.0.0 supports Python 3.13.
SNAP_VERSION="$PYTHON_VERSION_MINOR.0.0"

# Download SNAP to install directory and quietly install it
info "Downloading and installing SNAP $SNAP_VERSION..."

# WARNING: while file esa-snap_all_linux$SNAP_VERSION.sh regards the whole snap version
# description (e.g. 13.0.0), ${SNAP_VERSION%.*} regards the whole snap version
# description excluding the last part (e.g. 13.0).
if wget --content-disposition "https://download.esa.int/step/snap/${SNAP_VERSION%.*}/installers/esa-snap_all_linux-$SNAP_VERSION.sh"; then
    :
else
    error "Failed to download SNAP."
    cd "$WORKING_DIR"
    return 1
fi

if chmod +x "esa-snap_all_linux-$SNAP_VERSION.sh"; then
    :
else
    error "Failed to make SNAP installer executable."
    cd "$WORKING_DIR"
    return 1
fi

if "./esa-snap_all_linux-$SNAP_VERSION.sh" -q; then
    :
else
    error "Failed to install SNAP."
    cd "$WORKING_DIR"
    return 1
fi
if rm "esa-snap_all_linux-$SNAP_VERSION.sh"; then
    :
else
    error "Failed to remove SNAP installer."
    cd "$WORKING_DIR"
    return 1
fi

info "Done."

# Change to original working directory
cd "$WORKING_DIR"

# ---> Configure snappy

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