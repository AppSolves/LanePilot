#!/bin/bash
set -e

echo "🔧 Enabling UART and USB max current..."

CONFIG_FILE="/boot/firmware/config.txt"

# Insert settings under [all] section
if grep -q "^\[all\]" "$CONFIG_FILE"; then
  # Remove any existing lines for these settings under [all]
  sed -i '/^usb_max_current_enable=1/d' "$CONFIG_FILE"
  sed -i '/^dtparam=uart0=on/d' "$CONFIG_FILE"
  # Insert after [all]
  awk '/^\[all\]/{print;print "usb_max_current_enable=1\ndtparam=uart0=on";next}1' "$CONFIG_FILE" > /tmp/config.txt && sudo mv /tmp/config.txt "$CONFIG_FILE"
else
  # Add [all] section and settings at the end
  echo -e "\n[all]\nusb_max_current_enable=1\ndtparam=uart0=on" | sudo tee -a "$CONFIG_FILE"
fi

# Disable serial console (optional)
sudo raspi-config nonint do_serial 1 0

echo "✅ UART and USB max current enabled. Please reboot the Pi."