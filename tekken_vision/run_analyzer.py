import cv2
import numpy as np
import csv
import os
import sys
import json
import glob
import math
import argparse
from tqdm import tqdm

# It's good practice to handle the possibility of missing libraries
try:
    import easyocr
    import torch
except ImportError:
    print("Error: Required libraries not found. Please install them:")
    print("pip install easyocr torch torchvision torchaudio")
    sys.exit(1)

# --- Add project root to Python path ---
# This allows us to import from our 'tekken_vision' module
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
# -------------------------------------

from tekken_vision.health_detector import get_health_calibrated

# ==========================================================
# 1. Configuration & Global Setup
# ==========================================================
CONFIG_FILE_PATH = os.path.join(project_root, 'config.json')
VIDEO_INPUT_DIR = os.path.join(project_root, 'data', '01_raw_videos')
CSV_OUTPUT_DIR = os.path.join(project_root, 'data', '03_processed_csVs', 'vision_output')

def load_config():
    """Loads the configuration from the JSON file."""
    if not os.path.exists(CONFIG_FILE_PATH):
        print(f"FATAL ERROR: Configuration file not found at '{CONFIG_FILE_PATH}'")
        print("Please run 'python tools/run_setup.py' first to generate it.")
        return None
    
    with open(CONFIG_FILE_PATH, 'r') as f:
        config = json.load(f)
    print("Configuration loaded successfully.")
    return config

# ==========================================================
# 2. Helper Functions
# ==========================================================

def custom_round(number, places=2, threshold=0.5):
    """
    Rounds a number based on a variable threshold. For example, a threshold
    of 0.4 will round up from .4 onwards.
    """
    add_value = 1.0 - threshold
    multiplier = 10 ** places
    rounded_number = math.floor(float(number) * multiplier + add_value) / multiplier
    return f"{rounded_number:.{places}f}"

def get_timer_from_frame(frame, roi_coords, ocr_reader):
    """Extracts the integer timer value from a frame using OCR."""
    x, y, w, h = roi_coords
    timer_roi = frame[y:y+h, x:x+w]
    
    # Pre-processing for better OCR accuracy
    gray_roi = cv2.cvtColor(timer_roi, cv2.COLOR_BGR2GRAY)
    _, binary_roi = cv2.threshold(gray_roi, 180, 255, cv2.THRESH_BINARY)
    
    result = ocr_reader.readtext(binary_roi, allowlist='0123456789')
    if result:
        try:
            return int(result[0][1])
        except (ValueError, IndexError):
            return -1
    return -1

# ==========================================================
# 3. Analysis Modes
# ==========================================================

def run_full_analysis(video_path, config, ocr_reader):
    """
    Performs the detailed frame-by-frame analysis for health and timer,
    applies filtering, and generates the final data.
    """
    # --- Pass 1: Find Timer Anchors ---
    print("  > Pass 1: Finding timer anchor points...")
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    anchors = []
    last_timer_value = -1

    for frame_count in tqdm(range(0, total_frames, 2), desc="  Scanning for timer changes", unit="frame"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        ret, frame = cap.read()
        if not ret: break
        
        current_timer_value = get_timer_from_frame(frame, tuple(config['timer_roi']), ocr_reader)
        if current_timer_value != -1 and current_timer_value != last_timer_value:
            anchors.append((frame_count, current_timer_value))
            last_timer_value = current_timer_value
    
    cap.release()
    print(f"    Found {len(anchors)} timer anchor points.")

    if len(anchors) < 2:
        print("  Error: Not enough timer changes detected for full analysis. Skipping.")
        return None

    # --- Pass 2: Analyze Health and Interpolate Timer ---
    print("  > Pass 2: Analyzing health and interpolating timer...")
    cap = cv2.VideoCapture(video_path)
    
    # Unpack config values for easier access
    p1_coords = config['p1_coords']
    p2_coords = config['p2_coords']
    p1_max_pixels = config['p1_calibration']['max_pixels']
    p2_max_pixels = config['p2_calibration']['max_pixels']
    # --- UPDATED: Load separate color ranges ---
    p1_lower = np.array(config['p1_color_range']['lower'])
    p1_upper = np.array(config['p1_color_range']['upper'])
    p2_lower = np.array(config['p2_color_range']['lower'])
    p2_upper = np.array(config['p2_color_range']['upper'])
    CONFIRMATION_FRAMES = 10
    
    # Initialize health tracking state
    p1_last_confirmed, p1_potential, p1_counter = 1.0, 1.0, 0
    p2_last_confirmed, p2_potential, p2_counter = 1.0, 1.0, 0
    
    log_buffer = []
    final_data = []

    for frame_count in tqdm(range(total_frames), desc="  Analyzing frames", unit="frame"):
        ret, frame = cap.read()
        if not ret: break

        # --- UPDATED: Use correct color range for each player ---
        p1_raw = get_health_calibrated(frame, tuple(p1_coords['start']), tuple(p1_coords['end']), p1_lower, p1_upper, p1_max_pixels)
        p2_raw = get_health_calibrated(frame, tuple(p2_coords['start']), tuple(p2_coords['end']), p2_lower, p2_upper, p2_max_pixels)
        
        # Player 1 health filtering
        if abs(p1_raw - p1_potential) > 0.001:
            p1_potential = p1_raw; p1_counter = 1
        else:
            p1_counter += 1
        if p1_counter >= CONFIRMATION_FRAMES and p1_raw < p1_last_confirmed:
            p1_last_confirmed = p1_raw
            for i in range(1, min(len(log_buffer), CONFIRMATION_FRAMES) + 1):
                log_buffer[-i][1] = custom_round(p1_raw, threshold=0.4)

        # Player 2 health filtering
        if abs(p2_raw - p2_potential) > 0.001:
            p2_potential = p2_raw; p2_counter = 1
        else:
            p2_counter += 1
        if p2_counter >= CONFIRMATION_FRAMES and p2_raw < p2_last_confirmed:
            p2_last_confirmed = p2_raw
            for i in range(1, min(len(log_buffer), CONFIRMATION_FRAMES) + 1):
                log_buffer[-i][2] = custom_round(p2_raw, threshold=0.4)

        # Discrete timer calculation
        FPS, START_TIMER = 30, 40.0
        seconds_passed = frame_count // FPS
        normalized_timer = max(0, START_TIMER - seconds_passed) / START_TIMER

        # Format and buffer the row
        current_row = [
            frame_count, 
            custom_round(p1_last_confirmed, threshold=0.5), 
            custom_round(p2_last_confirmed, threshold=0.5), 
            f"{normalized_timer:.3f}"
        ]
        log_buffer.append(current_row)
        
        if len(log_buffer) > CONFIRMATION_FRAMES:
            final_data.append(log_buffer.pop(0))

    final_data.extend(log_buffer)
    cap.release()
    return final_data

def run_win_rate_analysis(video_path, config):
    """
    FAST MODE: Analyzes only the last frame of a video to determine the winner.
    """
    print("  > Fast Mode: Analyzing last frame for win rate...")
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None

    # Unpack config values
    p1_coords = config['p1_coords']
    p2_coords = config['p2_coords']
    p1_max_pixels = config['p1_calibration']['max_pixels']
    p2_max_pixels = config['p2_calibration']['max_pixels']
    # --- UPDATED: Load separate color ranges ---
    p1_lower = np.array(config['p1_color_range']['lower'])
    p1_upper = np.array(config['p1_color_range']['upper'])
    p2_lower = np.array(config['p2_color_range']['lower'])
    p2_upper = np.array(config['p2_color_range']['upper'])

    # --- UPDATED: Use correct color range for each player ---
    p1_health = get_health_calibrated(frame, tuple(p1_coords['start']), tuple(p1_coords['end']), p1_lower, p1_upper, p1_max_pixels)
    p2_health = get_health_calibrated(frame, tuple(p2_coords['start']), tuple(p2_coords['end']), p2_lower, p2_upper, p2_max_pixels)

    # The timer at the end is assumed to be 0
    return [[total_frames - 1, custom_round(p1_health, threshold=0.5), custom_round(p2_health, threshold=0.5), "0.000"]]

# ==========================================================
# 4. Main Execution Block
# ==========================================================

def main():
    """Main function to orchestrate the analysis of all videos."""
    parser = argparse.ArgumentParser(description="Tekken Vision Analysis Engine")
    parser.add_argument('--mode', type=str, default='full', choices=['full', 'win_rate'],
                        help="Analysis mode: 'full' for detailed analysis, 'win_rate' for fast outcome-only analysis.")
    args = parser.parse_args()

    print(f"--- Tekken Vision Analysis Engine (Mode: {args.mode}) ---")
    config = load_config()
    if config is None:
        return

    ocr_reader = None
    if args.mode == 'full':
        use_gpu = torch.cuda.is_available()
        print(f"\nInitializing EasyOCR reader (GPU: {use_gpu})...")
        ocr_reader = easyocr.Reader(['en'], gpu=use_gpu)
        print("EasyOCR reader initialized.")

    os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)
    video_files = glob.glob(os.path.join(VIDEO_INPUT_DIR, '*.avi')) + glob.glob(os.path.join(VIDEO_INPUT_DIR, '*.mp4'))
    
    if not video_files:
        print(f"\nWarning: No .avi or .mp4 files found in '{VIDEO_INPUT_DIR}'.")
        return

    print(f"\nFound {len(video_files)} video(s) to analyze.")
    for video_path in video_files:
        print(f"\n--- Analyzing: {os.path.basename(video_path)} ---")
        
        if args.mode == 'full':
            processed_data = run_full_analysis(video_path, config, ocr_reader)
        else: # win_rate mode
            processed_data = run_win_rate_analysis(video_path, config)
        
        if processed_data is None:
            print("  Skipping CSV generation for this video due to an error.")
            continue
            
        # Save results to a CSV file
        base_filename = os.path.splitext(os.path.basename(video_path))[0]
        output_csv_path = os.path.join(CSV_OUTPUT_DIR, f"{base_filename}.csv")
        
        header = ['Frame', 'P1_Health', 'P2_Health', 'Normalized_Game_Timer']
        with open(output_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(processed_data)
            
        print(f"  Success: Data saved to '{os.path.relpath(output_csv_path, project_root)}'")

    print("\n--- All Videos Processed ---")

if __name__ == '__main__':
    main()