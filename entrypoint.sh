#!/bin/bash

# Retrieve the network interface's name
DEVICE_NAME=$(ls /sys/class/net | grep ^e)

# Set static IP addresses for the containers
ip addr add ${DEVICE_STATIC_IP}/24 dev ${DEVICE_NAME}
ip link set ${DEVICE_NAME} up

# Start the Python main application
exec python3 -m src.main