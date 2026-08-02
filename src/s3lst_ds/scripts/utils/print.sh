#!/usr/bin/env bash

# Exit on error (-e), on use of undefined variables (-u), and if any command in a
# pipeline fails (-o pipefail).
# For details, see: https://linuxcommand.org/lc3_man_pages/seth.html
set -euo pipefail

# ---> Define print functions
# Define function for printing info messages
info() {
    printf "\033[36m[INFO]\033[0m %s\n" "$*"
}

# Define function for printing warning messages
warn() {
    printf "\033[33m[WARN]\033[0m %s\n" "$*"
}

# Define function for printing error messages
error() {
    printf "\033[31m[ERROR]\033[0m %s\n" "$*"
}

# ---> Define text style variables

# Colours
RED="\033[34;31m"
GREEN="\033[34;32m"
YELLOW="\033[34;33m"
BLUE="\033[34;34m"
CYAN="\033[34;36m"
# Bold
BOLD="\033[1m"
# Reset
RESET="\033[0m"