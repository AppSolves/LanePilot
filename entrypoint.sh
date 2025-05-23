#!/bin/bash

set -e

# Determine the model file path
if [ -f /proc/device-tree/model ]; then
    MODEL_PATH="/proc/device-tree/model"
elif [ -f /host/device-model ]; then
    MODEL_PATH="/host/device-model"
else
    echo "[ENTRYPOINT] Error: No device model file found. Exiting." >&2
    exit 1
fi

# Check whether the script is running on a Raspberry Pi or Jetson
if grep -q "Raspberry Pi" /"$MODEL_PATH"; then
    echo "[ENTRYPOINT] Running on Raspberry Pi."
    MODEL_TYPE="Raspberry Pi"
elif grep -q "NVIDIA Jetson" "$MODEL_PATH"; then
    echo "[ENTRYPOINT] Running on NVIDIA Jetson."
    MODEL_TYPE="NVIDIA Jetson"
else
    echo "[ENTRYPOINT] Error: Unknown device type. Exiting." >&2
    exit 1
fi

# Configure the ethernet interface
source "$(dirname "$0")/config/ethernet_interface.sh"
# Start the hotspot if applicable
source "$(dirname "$0")/config/hotspot_ap.sh"

# Start the main application as specified user (default: appuser)
APP_USER="${1:-appuser}"
shift
echo "[ENTRYPOINT] Starting main application as user: $APP_USER..."
if command -v gosu >/dev/null 2>&1; then
    exec gosu "$APP_USER" "$@"
elif command -v sudo >/dev/null 2>&1; then
    echo "[ENTRYPOINT] gosu not found, falling back to sudo."
    exec sudo -E -u "$APP_USER" "$@"
else
    echo "[ENTRYPOINT] Error: Neither 'gosu' nor 'sudo' found. Cannot start application as $APP_USER." >&2
    exit 1
fi