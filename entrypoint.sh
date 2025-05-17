#!/bin/bash

# Set static IP addresses for the containers
ip addr add ${DEVICE_STATIC_IP}/24 dev eth0
ip link set eth0 up

# Start the Python main application
exec python3 -m src.main