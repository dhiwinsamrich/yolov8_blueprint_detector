#!/bin/bash

# Path to your image
IMAGE_PATH="D:\Handson Experience\blueprint-detection\datasets\Complete Images\Images\4.png"

# API endpoint
URL="http://127.0.0.1:8000/detect"

# Confidence threshold and return image flag
CONFIDENCE="0.25"
RETURN_IMAGE="true"

# Make POST request
curl -X POST "$URL" \
  -F "file=@$IMAGE_PATH" \
  -F "conf=$CONFIDENCE" \
  -F "return_image=$RETURN_IMAGE"
