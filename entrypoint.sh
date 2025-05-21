#!/bin/bash

set -e

# Check whether the script is running on a Raspberry Pi or Jetson
if grep -q "Raspberry Pi" /proc/device-tree/model; then
    echo "[ENTRYPOINT] Running on Raspberry Pi."
    MODEL_TYPE="Raspberry Pi"
elif grep -q "NVIDIA Jetson" /proc/device-tree/model; then
    echo "[ENTRYPOINT] Running on NVIDIA Jetson."
    MODEL_TYPE="NVIDIA Jetson"
else
    echo "[ENTRYPOINT] Error: Unknown device type. Exiting." >&2
    exit 1
fi

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

# Try starting a hotspot if the device is a Raspberry Pi via nmcli
if [ "$MODEL_TYPE" == "Raspberry Pi" ]; then
    # Check if UART is enabled
    UART_ENABLED=false
    if grep -q "^\[all\]" /boot/firmware/config.txt && \
       grep -q "^dtparam=uart0=on" /boot/firmware/config.txt; then
        UART_ENABLED=true
    fi

    if [ "$UART_ENABLED" = true ]; then
        echo "[ENTRYPOINT] UART is enabled."
    else
        echo "[ENTRYPOINT] Error: UART is NOT enabled in /boot/firmware/config.txt."
        echo "[ENTRYPOINT] Please enable UART via \`scripts/enable_uart.sh\` and reboot the device."
        exit 1
    fi

    # Retrieve the wifi network interface's name(s)
    WIFI_INTERFACES=($(ls /sys/class/net | grep ^w))
    if [ ${#WIFI_INTERFACES[@]} -eq 0 ]; then
        echo "[ENTRYPOINT] Warning: No wifi interfaces found. Skipping hotspot setup."
    else
        WIFI_DEVICE_NAME=${WIFI_INTERFACES[0]}
        if [ -z "$HOTSPOT_SSID" ] || [ -z "$HOTSPOT_PASSWORD" ]; then
            echo "[ENTRYPOINT] Warning: HOTSPOT_SSID or HOTSPOT_PASSWORD not set. Skipping hotspot setup."
        else
            if ! nmcli device wifi hotspot ifname ${WIFI_DEVICE_NAME} ssid ${HOTSPOT_SSID} password ${HOTSPOT_PASSWORD}; then
                echo "[ENTRYPOINT] Error: Failed to start hotspot." >&2
                exit 1
            fi
            echo "[ENTRYPOINT] Hotspot started on ${WIFI_DEVICE_NAME} with SSID ${HOTSPOT_SSID}."

            # Configure and start dnsmasq to redirect all DNS queries to DEVICE_STATIC_IP
            echo "address=/#/${DEVICE_STATIC_IP}" > /tmp/dnsmasq-hotspot.conf
            if pgrep dnsmasq >/dev/null 2>&1; then
                echo "[ENTRYPOINT] Stopping existing dnsmasq instance."
                pkill dnsmasq
            fi
            dnsmasq --no-daemon --conf-file=/tmp/dnsmasq-hotspot.conf &
            DNSMASQ_PID=$!
            trap "echo '[ENTRYPOINT] Stopping dnsmasq...'; kill $DNSMASQ_PID 2>/dev/null" EXIT
            echo "[ENTRYPOINT] dnsmasq started, redirecting all DNS to ${DEVICE_STATIC_IP}."
        fi
    fi
else
    echo "[ENTRYPOINT] Skipping hotspot setup for ${MODEL_TYPE}."
fi

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