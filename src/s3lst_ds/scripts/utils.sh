#!/usr/bin/env bash

# Functions for printing,

# ---> Define printing functions
# Define function for printing info messages
info() {
    printf "\033[36m[INFO]\033[0m %b\n" "$*"
}

# Define function for printing warning messages
warn() {
    printf "\033[33m[WARN]\033[0m %b\n" "$*"
}

# Define function for printing error messages
error() {
    printf "\033[31m[ERROR]\033[0m %b\n" "$*"
}

# ---> Define text style variables

# Colours
RED="\033[34;31m"
GREEN="\033[34;32m"
YELLOW="\033[34;33m"
BLUE="\033[34;34m"
CYAN="\033[34;36m"
MAGENTA="\033[35m"
# Bold
BOLD="\033[1m"
# Reset
RESET="\033[0m"