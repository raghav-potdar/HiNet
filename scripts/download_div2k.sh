#!/usr/bin/env bash
# Download DIV2K dataset (train HR + validation HR) into datasets/
set -euo pipefail

DATASET_DIR="${1:-datasets}"
mkdir -p "$DATASET_DIR"

TRAIN_URL="http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
VALID_URL="http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"

echo "============================================"
echo "  DIV2K Dataset Downloader"
echo "============================================"

# --- Train set ---
if [ -d "$DATASET_DIR/DIV2K_train_HR" ]; then
    COUNT=$(find "$DATASET_DIR/DIV2K_train_HR" -maxdepth 1 -name '*.png' | wc -l)
    echo "[Train] Already exists with $COUNT images — skipping download"
else
    echo "[Train] Downloading DIV2K_train_HR.zip ..."
    wget -q --show-progress -O "$DATASET_DIR/DIV2K_train_HR.zip" "$TRAIN_URL"
    echo "[Train] Extracting ..."
    unzip -q "$DATASET_DIR/DIV2K_train_HR.zip" -d "$DATASET_DIR"
    rm "$DATASET_DIR/DIV2K_train_HR.zip"
    COUNT=$(find "$DATASET_DIR/DIV2K_train_HR" -maxdepth 1 -name '*.png' | wc -l)
    echo "[Train] Done — $COUNT images"
fi

# --- Validation set ---
if [ -d "$DATASET_DIR/DIV2K_valid_HR" ]; then
    COUNT=$(find "$DATASET_DIR/DIV2K_valid_HR" -maxdepth 1 -name '*.png' | wc -l)
    echo "[Valid] Already exists with $COUNT images — skipping download"
else
    echo "[Valid] Downloading DIV2K_valid_HR.zip ..."
    wget -q --show-progress -O "$DATASET_DIR/DIV2K_valid_HR.zip" "$VALID_URL"
    echo "[Valid] Extracting ..."
    unzip -q "$DATASET_DIR/DIV2K_valid_HR.zip" -d "$DATASET_DIR"
    rm "$DATASET_DIR/DIV2K_valid_HR.zip"
    COUNT=$(find "$DATASET_DIR/DIV2K_valid_HR" -maxdepth 1 -name '*.png' | wc -l)
    echo "[Valid] Done — $COUNT images"
fi

echo "============================================"
echo "  Dataset ready at: $DATASET_DIR/"
echo "    Train: $DATASET_DIR/DIV2K_train_HR/"
echo "    Valid: $DATASET_DIR/DIV2K_valid_HR/"
echo "============================================"
