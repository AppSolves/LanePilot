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

if [ $? -ne 0 ]; then
  echo "❌ Error: Failed to build the OpenCV base image."
  exit 1
else
  echo "✅ OpenCV base image built successfully."
  echo "🚀 Pushing the OpenCV base image to ghcr.io/appsolves/lanepilot/opencv_base:latest"
  
  # Check if the video_codec_sdk folder is not empty
  if [ -d "video_codec_sdk" ] && [ "$(ls -A video_codec_sdk)" ]; then
    echo "❌ Error: video_codec_sdk folder is not empty. Pushing the image would violate NVIDIA's EULA."
    exit 1
  fi

  docker push ghcr.io/appsolves/lanepilot/opencv_base:latest

  if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to push the OpenCV base image."
    exit 1
  else
    echo "✅ OpenCV base image pushed successfully."
  fi
fi