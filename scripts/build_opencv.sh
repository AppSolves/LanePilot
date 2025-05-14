#!/bin/bash

set -e

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
  echo "❌ Error: Docker is not running. Please start Docker and try again."
  exit 1
fi

cd "$(dirname "$0")/../opencv" || exit 1

docker buildx build --platform linux/arm64 --build-context root=../ -f Dockerfile.opencv \
    -t ghcr.io/appsolves/lanepilot/opencv_base:latest .

echo "✅ Done. OpenCV base image built successfully."