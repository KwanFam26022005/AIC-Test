#!/bin/bash

# Dataset path
DATASET_DIR="/tmp2/maitanha/vgu/ttn/data/AIC2025"

echo "=== Starting Dataset Restructuring ==="
echo "Dataset directory: $DATASET_DIR"

if [ ! -d "$DATASET_DIR" ]; then
    echo "ERROR: Dataset directory $DATASET_DIR does not exist!"
    exit 1
fi

# Create target parent folders
PARENT_KEYFRAMES="$DATASET_DIR/keyframes"
PARENT_VIDEOS="$DATASET_DIR/videos"

mkdir -p "$PARENT_KEYFRAMES"
mkdir -p "$PARENT_VIDEOS"

# 1. Move Keyframes_L* folders into parent keyframes folder, flattening their nested keyframes/ subdirectories
echo ""
echo "--- Restructuring Keyframes_L* ---"
for kf_dir in "$DATASET_DIR"/Keyframes_L[0-9]*; do
    if [ -d "$kf_dir" ] && [ "$(basename "$kf_dir")" != "keyframes" ]; then
        dirname=$(basename "$kf_dir")
        target_kf_dir="$PARENT_KEYFRAMES/$dirname"
        
        echo "Processing $dirname -> $target_kf_dir"
        mkdir -p "$target_kf_dir"
        
        # If the nested 'keyframes' directory exists, move its contents to target_kf_dir
        if [ -d "$kf_dir/keyframes" ]; then
            find "$kf_dir/keyframes" -maxdepth 1 -mindepth 1 | while read -r item; do
                mv "$item" "$target_kf_dir/"
            done
            rmdir "$kf_dir/keyframes" 2>/dev/null
        fi
        
        # Move any other files from kf_dir to target_kf_dir
        find "$kf_dir" -maxdepth 1 -mindepth 1 | while read -r item; do
            mv "$item" "$target_kf_dir/"
        done
        
        # Remove old directory
        rmdir "$kf_dir" 2>/dev/null
    fi
done

# 2. Move Videos_L* folders into parent videos folder, flattening their nested video/ subdirectories
echo ""
echo "--- Restructuring Videos_L* ---"
for vid_dir in "$DATASET_DIR"/Videos_L[0-9]*; do
    if [ -d "$vid_dir" ] && [ "$(basename "$vid_dir")" != "videos" ]; then
        dirname=$(basename "$vid_dir")
        target_vid_dir="$PARENT_VIDEOS/$dirname"
        
        echo "Processing $dirname -> $target_vid_dir"
        mkdir -p "$target_vid_dir"
        
        # If the nested 'video' directory exists, move its contents to target_vid_dir
        if [ -d "$vid_dir/video" ]; then
            find "$vid_dir/video" -maxdepth 1 -mindepth 1 | while read -r item; do
                mv "$item" "$target_vid_dir/"
            done
            rmdir "$vid_dir/video" 2>/dev/null
        fi
        
        # Move any other files from vid_dir to target_vid_dir
        find "$vid_dir" -maxdepth 1 -mindepth 1 | while read -r item; do
            mv "$item" "$target_vid_dir/"
        done
        
        # Remove old directory
        rmdir "$vid_dir" 2>/dev/null
    fi
done

echo ""
echo "=== Restructuring Completed! ==="
