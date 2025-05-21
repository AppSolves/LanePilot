#!/bin/bash

set -e

# Check whether the script is running on a Raspberry Pi or Jetson
if grep -q "Raspberry Pi" /proc/device-tree/model; then
    echo "[INTERFACE CONFIG] Running on Raspberry Pi."
    MODEL_TYPE="Raspberry Pi"
elif grep -q "NVIDIA Jetson" /proc/device-tree/model; then
    echo "[INTERFACE CONFIG] Running on NVIDIA Jetson."
    MODEL_TYPE="NVIDIA Jetson"
else
    echo "[INTERFACE CONFIG] Error: Unknown device type. Exiting." >&2
    exit 1
fi

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

ETH_DEVICE_NAME=${ETH_INTERFACES[0]}

# Set static IP addresses for the containers
if ip addr replace ${DEVICE_STATIC_IP}/24 dev ${ETH_DEVICE_NAME} && ip link set ${ETH_DEVICE_NAME} up; then
    echo "[INTERFACE CONFIG] Static IP address ${DEVICE_STATIC_IP} set for interface ${ETH_DEVICE_NAME}."
else
    echo "[INTERFACE CONFIG] Error: Failed to configure interface ${ETH_DEVICE_NAME}." >&2
    exit 1
fi

# Try starting a hotspot if the device is a Raspberry Pi via nmcli
if [ "$MODEL_TYPE" == "Raspberry Pi" ]; then
    # Retrieve the wifi network interface's name(s)
    WIFI_INTERFACES=($(ls /sys/class/net | grep ^w))
    if [ ${#WIFI_INTERFACES[@]} -eq 0 ]; then
        echo "[INTERFACE CONFIG] Warning: No wifi interfaces found. Skipping hotspot setup."
    else
        WIFI_DEVICE_NAME=${WIFI_INTERFACES[0]}
        if [ -z "$HOTSPOT_SSID" ] || [ -z "$HOTSPOT_PASSWORD" ]; then
            echo "[INTERFACE CONFIG] Warning: HOTSPOT_SSID or HOTSPOT_PASSWORD not set. Skipping hotspot setup."
        else
            if ! nmcli device wifi hotspot ifname ${WIFI_DEVICE_NAME} ssid ${HOTSPOT_SSID} password ${HOTSPOT_PASSWORD}; then
                echo "[INTERFACE CONFIG] Error: Failed to start hotspot." >&2
                exit 1
            fi
        fi
    fi
else
    echo "[INTERFACE CONFIG] Skipping hotspot setup for ${MODEL_TYPE}."
fi

# Start the main application as "appuser"
echo "[INTERFACE CONFIG] Starting main application..."
exec gosu appuser "$@"