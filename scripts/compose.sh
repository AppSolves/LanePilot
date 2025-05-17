#!/bin/bash

set -e

if [[ "$1" != "raspberrypi" && "$1" != "jetson" ]]; then
  echo "Usage: $0 [raspberrypi|jetson]"
  exit 1
fi

# Set the URL for the remote compose.yaml
COMPOSE_URL="https://raw.githubusercontent.com/AppSolves/LanePilot/refs/heads/main/firmware/$1/compose.yaml"

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

echo "Pulling the latest image for $1..."
curl -fsSL "$COMPOSE_URL" | docker compose -f - pull

echo "Starting (or restarting) the $1 container..."
curl -fsSL "$COMPOSE_URL" | docker compose -f - up -d

echo "✅ Done. The $1 container is running with the latest image."