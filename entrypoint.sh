#!/bin/bash

# Set the default IP address for the ethernet interface
if [ -z "$DEVICE_STATIC_IP" ]; then
    DEVICE_STATIC_IP="0.0.0.0"
    echo "[INTERFACE CONFIG] Warning: DEVICE_STATIC_IP is not set. Defaulting to 0.0.0.0."
fi

# Retrieve the ethernet network interface's name(s)
ETH_INTERFACES=($(ls /sys/class/net | grep ^e))

if [ ${#ETH_INTERFACES[@]} -eq 0 ]; then
    echo "[INTERFACE CONFIG] Error: No ethernet interfaces found." >&2
    exit 1
fi

DEVICE_NAME=${ETH_INTERFACES[0]}

# Set static IP addresses for the containers
if ip addr replace ${DEVICE_STATIC_IP}/24 dev ${DEVICE_NAME} && ip link set ${DEVICE_NAME} up; then
    echo "[INTERFACE CONFIG] Static IP address ${DEVICE_STATIC_IP} set for interface ${DEVICE_NAME}."
else
    echo "[INTERFACE CONFIG] Error: Failed to configure interface ${DEVICE_NAME}." >&2
    exit 1
fi

# Start the main application as "appuser"
echo "[INTERFACE CONFIG] Starting main application..."
exec gosu appuser "$@"