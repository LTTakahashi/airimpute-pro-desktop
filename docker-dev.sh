#!/bin/bash
# Development script for running AirImpute Pro in Docker with GUI support

set -e

echo "Starting AirImpute Pro in Docker development container..."

# Allow X11 connections from Docker
xhost +local:docker 2>/dev/null || true

# Build the development image
echo "Building Docker image..."
docker-compose build dev

# Run the development container with X11 forwarding
echo "Starting development container..."
docker-compose run --rm \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v $HOME/.Xauthority:/root/.Xauthority:rw \
    --privileged \
    dev

# Cleanup X11 permissions
xhost -local:docker 2>/dev/null || true