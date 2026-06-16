import csv
import os
import sys
from pathlib import Path
import subprocess
import cv2
import numpy as np
import time
import shutil
import logging

# Set up logging
def get_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("keyframe_extractor")

logger = get_logger()

# Append project root to import load_all_video_keyframes_info
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.load_all_video_keyframes_info import load_all_video_keyframes_info

# Define data paths
DATASET_DIR = "/tmp2/maitanha/vgu/ttn/data/AIC2025"
all_video, video_keyframe_dict = load_all_video_keyframes_info(DATASET_DIR)

def detect_video_codec(video_path):
    """Detect video codec using ffprobe"""
    try:
        cmd = ["ffprobe", "-v", "quiet", "-select_streams", "v:0", 
               "-show_entries", "stream=codec_name", "-of", "csv=p=0", video_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception as e:
        logger.warning(f"ffprobe failed for {video_path}: {e}")
    return "unknown"

def get_video_fps(video_path):
    """Get video FPS using ffprobe"""
    try:
        cmd = ["ffprobe", "-v", "quiet", "-select_streams", "v:0", 
               "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", video_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            fps_str = result.stdout.strip()
            if '/' in fps_str:
                num, den = fps_str.split('/')
                fps = float(num) / float(den) if float(den) != 0 else 25
            else:
                fps = float(fps_str) if fps_str else 25
            return fps
    except Exception as e:
        logger.warning(f"Failed to get FPS for {video_path}: {e}")
        return 25  # default fps
    
def keyframe_extractor(v):
    prefix = v.split("_")[0]
    
    # Paths configured according to instructions
    video_path = os.path.join(DATASET_DIR, "videos", f"Videos_{prefix}", f"{v}.mp4")
    file_path = os.path.join(DATASET_DIR, "transnetv2", "scenes", f"{v}.scenes.txt")
    
    kf_path = os.path.join(DATASET_DIR, "keyframes_transnetv2", f"Keyframes_{prefix}", v)
    map_path = os.path.join(DATASET_DIR, "map_keyframes", f"{v}.csv")

    if not os.path.exists(video_path):
        logger.warning(f"Video file {video_path} does not exist! Skipping.")
        return

    if not os.path.exists(file_path):
        logger.warning(f"Transnetv2 scenes file {file_path} does not exist! Skipping.")
        return

    if Path(kf_path).is_dir():
        logger.info(f"{kf_path} exists, skipping keyframe extraction...")
        return

    os.makedirs(kf_path, exist_ok=True)

    # Check codec
    codec = detect_video_codec(video_path)
    logger.info(f"Video {v} codec: {codec}")
    
    # Default to OpenCV with optimized smart-seeking to avoid FFmpeg command parsing & driver issues
    use_ffmpeg = False
    fps = get_video_fps(video_path)

    cap = None
    if not use_ffmpeg:
        cap = cv2.VideoCapture(video_path)

    with open(map_path, "w") as mapping_file, open(file_path, "r") as file:
        mapping = csv.writer(mapping_file)
        mapping.writerow(["n", "pts_time", "fps", "frame_idx"])
        lines = file.readlines()
        
        # If the file is empty, there is nothing to extract
        if not lines:
            logger.warning(f"Scenes file {file_path} is empty. No frames to extract.")
            if cap is not None:
                cap.release()
            return

        if use_ffmpeg:
            logger.info(f"Using single-call batch extraction for {codec} codec")
            start_time = time.time()
            
            # Prepare all frame numbers and output paths
            frame_numbers = []
            output_paths = []
            for i, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                left, right = parts[0], parts[1]
                mid = (int(left) + int(right)) // 2
                frame_numbers.append(mid)
                output_paths.append(os.path.join(kf_path, f"{i+1:04}.jpg"))
            
            if not frame_numbers:
                logger.warning(f"No valid frames parsed for {v}.")
                return

            # Create a single filter that extracts all frames at once
            frame_selects = "+".join([f"eq(n\\,{fn})" for fn in frame_numbers])
            
            # Build the GPU-accelerated ffmpeg command
            if codec == "av1":
                cmd = [
                    "ffmpeg",
                    "-c:v", "av1_cuvid",
                    "-i", video_path,
                    "-vf", f"select={frame_selects}",
                    "-vsync", "0",
                    "-y", "-loglevel", "quiet"
                ]
            elif codec == "h264":
                cmd = [
                    "ffmpeg", 
                    "-c:v", "h264_cuvid",
                    "-i", video_path,
                    "-vf", f"select={frame_selects}",
                    "-vsync", "0",
                    "-y", "-loglevel", "quiet"
                ]
            elif codec == "hevc":
                cmd = [
                    "ffmpeg",
                    "-c:v", "hevc_cuvid", 
                    "-i", video_path,
                    "-vf", f"select={frame_selects}",
                    "-vsync", "0",
                    "-y", "-loglevel", "quiet"
                ]
            else:
                cmd = [
                    "ffmpeg",
                    "-hwaccel", "cuda",
                    "-i", video_path,
                    "-vf", f"select={frame_selects}", 
                    "-vsync", "0",
                    "-y", "-loglevel", "quiet"
                ]
            
            # Use a temporary directory for batch output
            temp_dir = f"/tmp/batch_extract_{v}"
            os.makedirs(temp_dir, exist_ok=True)
            cmd.append(f"{temp_dir}/frame_%04d.jpg")
            
            logger.info(f"Extracting {len(frame_numbers)} frames in single GPU call...")
            
            try:
                result = subprocess.run(cmd, capture_output=True, timeout=300)  # 5 min timeout
                
                all_results = []
                if result.returncode == 0:
                    # Move files to correct locations
                    for i, (frame_num, output_path) in enumerate(zip(frame_numbers, output_paths)):
                        temp_file = f"{temp_dir}/frame_{i+1:04d}.jpg"
                        if os.path.exists(temp_file):
                            try:
                                os.rename(temp_file, output_path)
                                all_results.append((i + 1, frame_num, True, output_path))
                            except Exception as e:
                                logger.warning(f"Failed to move {temp_file}: {e}")
                                all_results.append((i + 1, frame_num, False, output_path))
                        else:
                            all_results.append((i + 1, frame_num, False, output_path))
                else:
                    # Fallback to single-call CPU batch extraction (much faster than sequential calls)
                    logger.error(f"Batch GPU extraction failed, falling back to CPU batch extraction")
                    cpu_cmd = [
                        "ffmpeg",
                        "-i", video_path,
                        "-vf", f"select={frame_selects}",
                        "-vsync", "0",
                        "-y", "-loglevel", "quiet",
                        f"{temp_dir}/frame_%04d.jpg"
                    ]
                    cpu_result = subprocess.run(cpu_cmd, capture_output=True)
                    
                    all_results = []
                    for i, (frame_num, output_path) in enumerate(zip(frame_numbers, output_paths)):
                        temp_file = f"{temp_dir}/frame_{i+1:04d}.jpg"
                        if cpu_result.returncode == 0 and os.path.exists(temp_file):
                            try:
                                os.rename(temp_file, output_path)
                                all_results.append((i + 1, frame_num, True, output_path))
                            except Exception as e:
                                logger.warning(f"Failed to move {temp_file}: {e}")
                                all_results.append((i + 1, frame_num, False, output_path))
                        else:
                            all_results.append((i + 1, frame_num, False, output_path))
                
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    
            except subprocess.TimeoutExpired:
                logger.error("Batch extraction timed out")
                all_results = [(i + 1, fn, False, op) for i, (fn, op) in enumerate(zip(frame_numbers, output_paths))]
            
            total_time = time.time() - start_time
            successful_frames = sum(1 for r in all_results if r[2])
            
            # Write results to mapping file in order
            for idx, frame_num, success, keyframe_path in all_results:
                if success:
                    mapping.writerow([idx, round(frame_num / fps, 4), fps, frame_num])
                    print(f"Saved keyframe {frame_num} of video {v} using single-call ffmpeg")
                else:
                    print(f"Failed to extract frame {frame_num} from video {v} using ffmpeg.", file=sys.stderr)
            
            logger.info(f"{successful_frames}/{len(lines)} frames in {total_time:.2f}s ({successful_frames/total_time:.1f} fps)")
            
        else:
            # Use OpenCV optimized single-pass sequential reading
            logger.info("Using optimized single-pass OpenCV extraction")
            
            # Prepare all frame numbers and output paths
            frame_numbers = []
            output_paths = []
            for i, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                left, right = parts[0], parts[1]
                mid = (int(left) + int(right)) // 2
                frame_numbers.append(mid)
                output_paths.append(os.path.join(kf_path, f"{i+1:04}.jpg"))
                
            sorted_indices = sorted(range(len(frame_numbers)), key=lambda k: frame_numbers[k])
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 25.0
                
            current_frame = 0
            for target_idx in sorted_indices:
                target_frame = frame_numbers[target_idx]
                while current_frame <= target_frame:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if current_frame == target_frame:
                        output_path = output_paths[target_idx]
                        cv2.imwrite(output_path, frame)
                        mapping.writerow([target_idx + 1, round(target_frame / fps, 4), fps, target_frame])
                        print(f"Saved keyframe {target_frame} of video {v} using OpenCV (single-pass)")
                    current_frame += 1

    if cap is not None:
        cap.release()


if __name__ == "__main__":
    os.makedirs(os.path.join(DATASET_DIR, "map_keyframes"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "keyframes_transnetv2"), exist_ok=True)

    logger.info("Starting keyframe extraction process")
    for v in all_video:
        start = time.time()
        keyframe_extractor(v)
        end = time.time()
        logger.info(f"Processed video {v} in {end - start:.2f} seconds")
