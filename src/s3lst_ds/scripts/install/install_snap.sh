#!/usr/bin/env bash

# Install SNAP and configure

# ---> Get helpful paths
WORKING_DIR="$PWD"
# Get absolute path to scripts directory
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Get relative path from working directory to this very script
SCRIPT_FILE_REL=$(realpath --relative-to="$WORKING_DIR" "${SCRIPTS_DIR}/install/install_snap.sh")
# Get absolute path to SNAP installation directory
SNAP_DIR="$HOME/esa-snap"

# ---> Get utility functions
source "$SCRIPTS_DIR/utils.sh" || return 1

# ---> Define script help, error message functions
# Function to display help
# NOTE: Help message may be prompted by using the command `source install.sh --help`
show_help() {
    echo Install SNAP and configure snappy.
    echo -e ""
    echo -e "${BOLD}${GREEN}Usage:${RESET} ${CYAN}source $SCRIPT_FILE_REL [OPTIONS]${RESET}"
    echo -e ""
    echo -e "${BOLD}${GREEN}Options:${RESET}"
    echo -e "      ${BOLD}${CYAN}--force${RESET}  Force reinstallation of SNAP even if already installed."
    echo -e "  ${BOLD}${CYAN}-h, --help${RESET}          Show this help message"
}

# Function to display error message for unknown options
error_unknown_option() {
    echo -e "${BOLD}${RED}error:${RESET} unknown option ${YELLOW}'$1'${RESET}"
    echo -e ""
    echo -e "${BOLD}${GREEN}Usage:${RESET} ${CYAN}source $SCRIPT_FILE_REL [OPTIONS]${RESET}"
    echo -e ""
    echo -e "Use ${BOLD}${CYAN}-h, --help${RESET} to see available options."
}


# ---> Parse script arguments
FORCE=0
for arg in "$@"; do
    case $arg in
        --force)
            FORCE=1
            shift
            ;;
        -h|--help)
            show_help
            return 0
            ;;
        *)
            error_unknown_option "$arg"
            return 1
            ;;
    esac
done


# ---> Check if SNAP is already installed
# Check if path to SNAP installation directory is readable, and if snappy configuration
# script exists and is executable
if [ -r "$SNAP_DIR" ] && [ -x "$SNAP_DIR/bin/snappy-conf" ]; then
    SNAP_IS_INSTALLED=1
else
    SNAP_IS_INSTALLED=0
fi

# Do not install and solely configure if SNAP is already installed and the `--force`
# flag is not provided
if [ "$SNAP_IS_INSTALLED" -eq 1 ] && [ "$FORCE" -eq 0 ]; then
    warn "SNAP is already installed and --force flag is not provided. Skipping reinstallation..."
# Uninstall SNAP dependencies if SNAP is already installed and the `--force` flag is
# provided
elif [ "$SNAP_IS_INSTALLED" -eq 1 ] && [ "$FORCE" -eq 1 ]; then
    warn "SNAP is already installed, but the --force flag is provided. Proceeding to reinstall SNAP..."

    source "$SCRIPTS_DIR/install/uninstall_snap.sh" || return 1
fi


# ---> Download and install SNAP if SNAP is not already installed or if the `--force`
# flag is provided
if [ "$SNAP_IS_INSTALLED" -eq 0 ] || [ "$FORCE" -eq 1 ]; then

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
fi

# ---> Configure snappy
source "$SCRIPTS_DIR/install/configure_snap.sh" || return 1

