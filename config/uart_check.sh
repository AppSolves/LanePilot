#!/bin/bash

set -e

# Prevent script from being executed directly
(return 0 2>/dev/null) || { echo "This script must be sourced, not executed."; exit 1; }

# Check if UART is enabled
if grep -q "^\[all\]" /boot/firmware/config.txt && \
   grep -q "^dtparam=uart0=on" /boot/firmware/config.txt; then
    echo "[ENTRYPOINT] UART is enabled."
else
    echo "[ENTRYPOINT] Error: UART is NOT enabled in /boot/firmware/config.txt."
    echo "[ENTRYPOINT] Please enable UART via \`scripts/enable_uart.sh\` and reboot the device."
    exit 1
fi