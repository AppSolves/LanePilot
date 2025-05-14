#!/bin/bash

set -e

if [[ "$1" != "raspberry" && "$1" != "jetson" ]]; then
  echo "Usage: $0 [raspberry|jetson]"
  exit 1
fi

# Change to the directory containing the compose.yaml
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")/firmware/$1"
cd "$PROJECT_ROOT"

echo "Pulling the latest image for $1..."
docker compose pull

echo "Starting (or restarting) the $1 container..."
docker compose up -d

echo "✅ Done. The $1 container is running with the latest image."