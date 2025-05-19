#!/bin/bash

set -e

NO_UPDATE=false

# Parse arguments
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --no-update|-nu)
      NO_UPDATE=true
      shift
      ;;
    raspberrypi|jetson)
      TARGET="$1"
      shift
      ;;
    *)
      echo "Usage: $0 [--no-update|-nu] [raspberrypi|jetson]"
      exit 1
      ;;
  esac
done

if [[ "$TARGET" != "raspberrypi" && "$TARGET" != "jetson" ]]; then
  echo "Usage: $0 [--no-update|-nu] [raspberrypi|jetson]"
  exit 1
fi

# Set the URL for the remote compose.yaml
COMPOSE_URL="https://raw.githubusercontent.com/AppSolves/LanePilot/refs/heads/main/firmware/$TARGET/compose.yaml"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
  echo "❌ Error: Docker is not running. Please start Docker and try again."
  exit 1
fi

# Test whether the URL is reachable
if ! curl -s --head "$COMPOSE_URL" | grep "200 OK" > /dev/null; then
  echo "❌ Error: Unable to reach the URL '$COMPOSE_URL'. Please check your internet connection or the URL."
  exit 1
fi

if [ "$NO_UPDATE" = false ]; then
  echo "Pulling the latest image for $TARGET..."
  curl -fsSL "$COMPOSE_URL" | docker compose -f - pull
fi

echo "Starting (or restarting) the $TARGET container..."
curl -fsSL "$COMPOSE_URL" | docker compose -f - up -d

echo "✅ Done. The $TARGET container is running with the latest image."