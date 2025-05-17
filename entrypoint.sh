#!/bin/bash

# Retrieve the ethernet network interface's name(s)
ETH_INTERFACES=($(ls /sys/class/net | grep ^e))

if [ ${#ETH_INTERFACES[@]} -eq 0 ]; then
    echo "[ENTRYPOINT] Error: No ethernet interfaces found." >&2
    exit 1
fi

DEVICE_NAME=${ETH_INTERFACES[0]}

# Set static IP addresses for the containers
ip addr add ${DEVICE_STATIC_IP}/24 dev ${DEVICE_NAME}
ip link set ${DEVICE_NAME} up

# Start the Python main application
exec python3 -m src.main