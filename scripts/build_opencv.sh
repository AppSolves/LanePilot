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

if [ $? -ne 0 ]; then
  echo "❌ Error: Failed to build the OpenCV base image."
  exit 1
else
  echo "✅ OpenCV base image built successfully."

  # Check if the video_codec_sdk folder is not empty
  if [ -d "video_codec_sdk" ] && [ "$(ls -A video_codec_sdk)" ]; then
    # Ask the user if they want to build the jetson image too
    read -p "Do you want to build the Jetson image as well? (y/n): " build_jetson
    if [[ ! "$build_jetson" =~ ^[Yy]$ ]]; then
      echo "🚫 Skipping Jetson image build."
      exit 0
    else
      echo "🚀 Building the Jetson image..."
      docker buildx build --pull --platform linux/arm64 --build-context root=../ --build-context models=../../assets/trained_models \
          -f Dockerfile.jetson \
          -t ghcr.io/appsolves/lanepilot/jetson:latest .
      if [ $? -ne 0 ]; then
        echo "❌ Error: Failed to build the Jetson image."
        exit 1
      else
        echo "✅ Jetson image built successfully."
        exit 0
      fi
    fi
  fi

  # Ask the user if they want to push the image
  read -p "Do you want to push the OpenCV base image to ghcr.io/appsolves/lanepilot/opencv_base:latest? (y/n): " push_image
  if [[ ! "$push_image" =~ ^[Yy]$ ]]; then
    echo "🚫 Skipping image push."
    exit 0
  fi

  echo "🚀 Pushing the OpenCV base image to ghcr.io/appsolves/lanepilot/opencv_base:latest"
  docker push ghcr.io/appsolves/lanepilot/opencv_base:latest

  if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to push the OpenCV base image."
    exit 1
  else
    echo "✅ OpenCV base image pushed successfully."
  fi
fi