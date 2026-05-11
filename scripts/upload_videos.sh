#!/bin/bash

set -a
source .env
set +a

DIRS=("profile_pics" "thumbnails" "videos")
DEST="${ONEDRIVE_VIDEOS_DEST}"

if [ -z "$DEST" ]; then
    echo "✗ ONEDRIVE_VIDEOS_DEST not set in .env"
    exit 1
fi

TOTAL_LOCAL=0
TOTAL_REMOTE=0

for DIR in "${DIRS[@]}"; do
    SRC="data/source/$DIR"
    echo "Uploading $DIR..."
    rclone copy "$SRC" "$DEST/$DIR" --progress

    LOCAL=$(find "$SRC" -type f | wc -l)
    REMOTE=$(rclone ls "$DEST/$DIR" | wc -l)

    echo "  Local: $LOCAL files | Remote: $REMOTE files"

    TOTAL_LOCAL=$((TOTAL_LOCAL + LOCAL))
    TOTAL_REMOTE=$((TOTAL_REMOTE + REMOTE))
done

echo ""
echo "Validating totals..."
echo "Total local: $TOTAL_LOCAL files"
echo "Total remote: $TOTAL_REMOTE files"

if [ "$TOTAL_LOCAL" -eq "$TOTAL_REMOTE" ]; then
    echo "✓ Validation passed"
else
    echo "✗ Mismatch — do not delete local files"
    exit 1
fi
