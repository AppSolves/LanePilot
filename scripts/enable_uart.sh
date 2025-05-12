#!/bin/bash
set -e

echo "🔧 Enabling UART..."

# Enable UART in /boot/config.txt
if ! grep -q "^enable_uart=1" /boot/config.txt; then
  echo "enable_uart=1" | sudo tee -a /boot/config.txt
fi

# Disable serial console (optional)
sudo raspi-config nonint do_serial 1 0

echo "✅ UART Enabled. Please reboot the Pi."
