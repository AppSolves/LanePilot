#!/bin/bash

set -e

# Prevent script from being executed directly
(return 0 2>/dev/null) || { echo "This script must be sourced, not executed."; exit 1; }

# Configure and start dnsmasq to redirect all DNS queries to DEVICE_STATIC_IP
if pgrep dnsmasq >/dev/null 2>&1; then
    echo "[ENTRYPOINT] Stopping existing dnsmasq instance."
    pkill dnsmasq
    while pgrep dnsmasq >/dev/null 2>&1; do
        sleep 0.1
    done
fi
dnsmasq --no-daemon --address=/#/${DEVICE_STATIC_IP} > /dev/null 2>&1 &
DNSMASQ_PID=$!
trap "echo '[ENTRYPOINT] Stopping dnsmasq...'; kill $DNSMASQ_PID 2>/dev/null" EXIT
echo "[ENTRYPOINT] dnsmasq started, redirecting all DNS to ${DEVICE_STATIC_IP}."