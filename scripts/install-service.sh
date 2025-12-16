#!/bin/bash
# Install ASRI scheduler as a systemd service

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVICE_FILE="$SCRIPT_DIR/asri-scheduler.service"

echo "Installing ASRI scheduler service..."

# Copy service file
sudo cp "$SERVICE_FILE" /etc/systemd/system/asri-scheduler.service

# Reload systemd
sudo systemctl daemon-reload

# Enable and start
sudo systemctl enable asri-scheduler
sudo systemctl start asri-scheduler

echo ""
echo "✅ Service installed and started"
echo ""
echo "Commands:"
echo "  sudo systemctl status asri-scheduler  # Check status"
echo "  sudo journalctl -u asri-scheduler -f  # View logs"
echo "  sudo systemctl restart asri-scheduler # Restart"
