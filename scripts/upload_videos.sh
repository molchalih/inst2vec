#!/bin/bash

set -a
source .env
set +a

SRC="data/source/videos"
DEST="${ONEDRIVE_VIDEOS_DEST}"

if [ -z "$DEST" ]; then
    echo "✗ ONEDRIVE_VIDEOS_DEST not set in .env"
    exit 1
fi

echo "Uploading videos..."
rclone copy "$SRC" "$DEST" --progress

echo "Validating..."
LOCAL=$(find "$SRC" -type f | wc -l)
REMOTE=$(rclone ls "$DEST" | wc -l)

echo "Local: $LOCAL files"
echo "Remote: $REMOTE files"

if [ "$LOCAL" -eq "$REMOTE" ]; then
    echo "✓ Validation passed"
else
    echo "✗ Mismatch — do not delete local files"
    exit 1
fi
