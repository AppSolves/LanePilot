#!/bin/bash

set -e

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
  echo "❌ Error: Docker is not running. Please start Docker and try again."
  exit 1
fi

cd "$(dirname "$0")/../opencv" || exit 1

docker buildx build --pull --platform linux/arm64 --build-context root=../ -f Dockerfile.opencv \
    -t ghcr.io/appsolves/lanepilot/opencv_base:latest .

echo "✅ OpenCV base image built successfully."

# Check if the video_codec_sdk folder is not empty
if [ -d "video_codec_sdk" ] && [ "$(ls -A video_codec_sdk)" ]; then
  read -r -p "Do you want to build the Jetson image as well? (y/n): " build_jetson
  if [[ "$build_jetson" =~ ^[Yy]$ ]]; then
    echo "🚀 Building the Jetson image..."
    cd "../firmware/jetson" || exit 1
    docker buildx build --pull --platform linux/arm64 --build-context root=../../ --build-context models=../../assets/trained_models \
        -f Dockerfile.jetson \
        -t ghcr.io/appsolves/lanepilot/jetson:latest .
    echo "✅ Jetson image built successfully."
  else
    echo "🚫 Skipping Jetson image build."
  fi
  exit 0
fi

read -r -p "Do you want to push the OpenCV base image to ghcr.io/appsolves/lanepilot/opencv_base:latest? (y/n): " push_image
if [[ "$push_image" =~ ^[Yy]$ ]]; then
  echo "🚀 Pushing the OpenCV base image to ghcr.io/appsolves/lanepilot/opencv_base:latest"
  docker push ghcr.io/appsolves/lanepilot/opencv_base:latest
  echo "✅ OpenCV base image pushed successfully."
else
  echo "🚫 Skipping image push."
fi