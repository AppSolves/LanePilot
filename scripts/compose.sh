#!/bin/bash

set -e

if [[ "$1" != "raspberry" && "$1" != "jetson" ]]; then
  echo "Usage: $0 [raspberry|jetson]"
  exit 1
fi

# Set the URL for the remote compose.yaml
COMPOSE_URL="https://raw.githubusercontent.com/AppSolves/LanePilot/refs/heads/main/firmware/$1/compose.yaml"

echo "Pulling the latest image for $1..."
curl -fsSL "$COMPOSE_URL" | docker compose -f - pull

echo "Starting (or restarting) the $1 container..."
curl -fsSL "$COMPOSE_URL" | docker compose -f - up -d

echo "✅ Done. The $1 container is running with the latest image."