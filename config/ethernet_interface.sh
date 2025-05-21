#!/bin/bash

set -e

# Prevent script from being executed directly
(return 0 2>/dev/null) || { echo "This script must be sourced, not executed."; exit 1; }

# Set the default IP address for the ethernet interface
if [ -z "$DEVICE_STATIC_IP" ]; then
    DEVICE_STATIC_IP="0.0.0.0"
    echo "[ENTRYPOINT] Warning: DEVICE_STATIC_IP is not set. Defaulting to 0.0.0.0."
fi

# Retrieve the ethernet network interface's name(s)
ETH_INTERFACES=($(ls /sys/class/net | grep ^e))

if [ ${#ETH_INTERFACES[@]} -eq 0 ]; then
    echo "[ENTRYPOINT] Error: No ethernet interfaces found." >&2
    exit 1
fi

ETH_DEVICE_NAME=${ETH_INTERFACES[0]}

# Set static IP addresses for the containers
if ip addr replace ${DEVICE_STATIC_IP}/24 dev ${ETH_DEVICE_NAME} && ip link set ${ETH_DEVICE_NAME} up; then
    echo "[ENTRYPOINT] Static IP address ${DEVICE_STATIC_IP} set for interface ${ETH_DEVICE_NAME}."
else
    echo "[ENTRYPOINT] Error: Failed to configure interface ${ETH_DEVICE_NAME}." >&2
    exit 1
fi