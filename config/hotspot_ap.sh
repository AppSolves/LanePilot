#!/bin/bash

set -e

# Prevent script from being executed directly
(return 0 2>/dev/null) || { echo "This script must be sourced, not executed."; exit 1; }

# Check if the script is being run on a Raspberry Pi
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

        if [ -z "$HOTSPOT_SSID" ] || [ -z "$HOTSPOT_PASSWORD" ] || [ -z "$HOTSPOT_IP" ]; then
            echo "[ENTRYPOINT] Warning: HOTSPOT_SSID, HOTSPOT_PASSWORD, or HOTSPOT_IP not set. Skipping hotspot setup."
        else
            if ! iw dev | grep -q "$AP_INTERFACE"; then
                iw dev "$WIFI_DEVICE_NAME" interface add "$AP_INTERFACE" type __ap
                echo "[ENTRYPOINT] Virtual AP interface $AP_INTERFACE created."
            else
                echo "[ENTRYPOINT] Virtual AP interface $AP_INTERFACE already exists. Recreating..."
                iw dev "$AP_INTERFACE" del
                iw dev "$WIFI_DEVICE_NAME" interface add "$AP_INTERFACE" type __ap
                echo "[ENTRYPOINT] Virtual AP interface $AP_INTERFACE recreated."
            fi

            ip link set "$AP_INTERFACE" up
            ip addr flush dev "$AP_INTERFACE"
            ip addr add "$HOTSPOT_IP/24" dev "$AP_INTERFACE"
            echo "[ENTRYPOINT] IP address $HOTSPOT_IP/24 set on $AP_INTERFACE."

            sleep 1

            echo "[ENTRYPOINT] Starting hostapd for $AP_INTERFACE without config file..."

            hostapd -B /dev/stdin <<EOF > /dev/null
interface=$AP_INTERFACE
driver=nl80211
ssid=$HOTSPOT_SSID
hw_mode=g
channel=6
wmm_enabled=0
macaddr_acl=0
auth_algs=1
ignore_broadcast_ssid=0
wpa=2
wpa_passphrase=$HOTSPOT_PASSWORD
wpa_key_mgmt=WPA-PSK
rsn_pairwise=CCMP
EOF

            if [ -f "$(dirname "$0")/config/dns_mask.sh" ]; then
                source "$(dirname "$0")/config/dns_mask.sh"
            else
                echo "[ENTRYPOINT] dnsmasq script not found. DNS server not started."
            fi

            echo "[ENTRYPOINT] Hotspot setup complete."
        fi
    fi
else
    echo "[ENTRYPOINT] Skipping hotspot setup for $MODEL_TYPE."
fi