#!/bin/bash

set -e

# Prevent script from being executed directly
(return 0 2>/dev/null) || { echo "This script must be sourced, not executed."; exit 1; }

# Try starting a hotspot if the device is a Raspberry Pi via nmcli
if [ "$MODEL_TYPE" == "Raspberry Pi" ]; then
    # Check if UART is enabled
    source "$(dirname "$0")/config/uart_check.sh"

    # Retrieve the wifi network interface's name(s)
    WIFI_INTERFACES=($(ls /sys/class/net | grep ^w))
    if [ ${#WIFI_INTERFACES[@]} -eq 0 ]; then
        echo "[ENTRYPOINT] Warning: No wifi interfaces found. Skipping hotspot setup."
    else
        WIFI_DEVICE_NAME=${WIFI_INTERFACES[0]}
        AP_INTERFACE="uap0"
        if [ -z "$HOTSPOT_IP" ] || [ -z "$HOTSPOT_SSID" ] || [ -z "$HOTSPOT_PASSWORD" ]; then
            echo "[ENTRYPOINT] Warning: HOTSPOT_IP, HOTSPOT_SSID or HOTSPOT_PASSWORD not set. Skipping hotspot setup."
        else
            if ! iw dev | grep -q ${AP_INTERFACE}; then
                iw dev ${WIFI_DEVICE_NAME} interface add ${AP_INTERFACE} type __ap
                echo "[ENTRYPOINT] Virtual AP interface ${AP_INTERFACE} created."
            else
                echo "[ENTRYPOINT] Virtual AP interface ${AP_INTERFACE} already exists."
            fi

            # Set static IP address for the hotspot interface
            if ip addr replace ${HOTSPOT_IP}/24 dev ${AP_INTERFACE} && ip link set ${AP_INTERFACE} up; then
                echo "[ENTRYPOINT] Static IP address ${HOTSPOT_IP} set for interface ${AP_INTERFACE}."
            else
                echo "[ENTRYPOINT] Error: Failed to configure interface ${AP_INTERFACE}." >&2
                exit 1
            fi

            nmcli device wifi hotspot ifname ${AP_INTERFACE} ssid ${HOTSPOT_SSID} password ${HOTSPOT_PASSWORD} > /dev/null
            echo "[ENTRYPOINT] Hotspot started on ${AP_INTERFACE} with SSID ${HOTSPOT_SSID}."

            # Start the dnsmasq service
            source "$(dirname "$0")/config/dns_mask.sh"
        fi
    fi
else
    echo "[ENTRYPOINT] Skipping hotspot setup for ${MODEL_TYPE}."
fi