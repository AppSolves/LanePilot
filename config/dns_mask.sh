#!/bin/bash

set -e

# Prevent script from being executed directly
(return 0 2>/dev/null) || { echo "This script must be sourced, not executed."; exit 1; }

# Redirect all incoming HTTP traffic on the AP interface to port $DISPLAY_SERVER_PORT
iptables -t nat -A PREROUTING -i $AP_INTERFACE -p tcp --dport 80 -j REDIRECT --to-port $DISPLAY_SERVER_PORT
iptables -t nat -A PREROUTING -i $AP_INTERFACE -p tcp --dport 443 -j REDIRECT --to-port $DISPLAY_SERVER_PORT

# Configure and start dnsmasq to redirect all DNS queries to HOTSPOT_IP
if pgrep dnsmasq >/dev/null; then
    echo "[ENTRYPOINT] Stopping existing dnsmasq instance."
    pkill dnsmasq
    while pgrep dnsmasq >/dev/null; do
        sleep 0.1
    done
fi

# Start dnsmasq for DHCP + DNS redirection (captive portal style)
dnsmasq \
    --interface=$AP_INTERFACE \
    --bind-interfaces \
    --except-interface=lo \
    --dhcp-range=${HOTSPOT_IP%.*}.10,${HOTSPOT_IP%.*}.50,12h \
    --address=/#/$HOTSPOT_IP \
    --no-resolv \
    --log-queries \
    --log-dhcp \
    > /dev/null &
DNSMASQ_PID=$!
trap "echo '[ENTRYPOINT] Stopping dnsmasq...'; kill $DNSMASQ_PID 2>/dev/null" EXIT

# Check if dnsmasq started successfully
if ! kill -0 $DNSMASQ_PID 2>/dev/null; then
    echo "[ERROR] Failed to start dnsmasq."
    return 1
fi

echo "[ENTRYPOINT] dnsmasq started with DHCP and DNS redirection to $HOTSPOT_IP."
echo "[ENTRYPOINT] dnsmasq PID: $DNSMASQ_PID"