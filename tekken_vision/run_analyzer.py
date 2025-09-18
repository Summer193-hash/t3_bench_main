import cv2
import numpy as np
import csv
import os
import sys
import json
import glob
import math
import argparse # NEW: Import argparse for command-line flags
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
# (custom_round and get_timer_from_frame functions are unchanged)

def custom_round(number, places=2, threshold=0.5):
    """
    Rounds a number based on a variable threshold.
    """
    add_value = 1.0 - threshold
    multiplier = 10 ** places
    rounded_number = math.floor(float(number) * multiplier + add_value) / multiplier
    return f"{rounded_number:.{places}f}"

def get_timer_from_frame(frame, roi_coords, ocr_reader):
    """Extracts the integer timer value from a frame using OCR."""
    x, y, w, h = roi_coords
    timer_roi = frame[y:y+h, x:x+w]
    gray_roi = cv2.cvtColor(timer_roi, cv2.COLOR_BGR2GRAY)
    _, binary_roi = cv2.threshold(gray_roi, 180, 255, cv2.THRESH_BINARY)
    result = ocr_reader.readtext(binary_roi, allowlist='0123456789')
    if result:
        try: return int(result[0][1])
        except (ValueError, IndexError): return -1
    return -1

# ==========================================================
# 3. Analysis Modes
# ==========================================================

def run_full_analysis(video_path, config, ocr_reader):
    """
    Runs the complete two-pass analysis to generate detailed, frame-by-frame data.
    """
    # --- Pass 1: Find Timer Anchors ---
    print("  > Pass 1: Finding timer anchor points...")
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    anchors = []
    last_timer_value = -1

    for frame_count in tqdm(range(0, total_frames, 2), desc="  Scanning for timer changes", unit="frame"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        ret, frame = cap.read();
        if not ret: break
        current_timer_value = get_timer_from_frame(frame, tuple(config['timer_roi']), ocr_reader)
        if current_timer_value != -1 and current_timer_value != last_timer_value:
            anchors.append((frame_count, current_timer_value))
            last_timer_value = current_timer_value
    cap.release()
    print(f"    Found {len(anchors)} timer anchor points.")
    if len(anchors) < 2:
        print("  Error: Not enough timer changes detected. Skipping detailed analysis.")
        return None

    # --- Pass 2: Detailed Frame-by-Frame Analysis ---
    print("  > Pass 2: Analyzing health and interpolating timer...")
    cap = cv2.VideoCapture(video_path)
    # Unpack config values
    p1_coords, p2_coords = config['p1_coords'], config['p2_coords']
    p1_max_pixels, p2_max_pixels = config['p1_calibration']['max_pixels'], config['p2_calibration']['max_pixels']
    lower, upper = np.array(config['color_range']['lower']), np.array(config['color_range']['upper'])
    CONFIRMATION_FRAMES = 10
    
    p1_last_confirmed, p1_potential, p1_counter = 1.0, 1.0, 0
    p2_last_confirmed, p2_potential, p2_counter = 1.0, 1.0, 0
    log_buffer, final_data = [], []

    for frame_count in tqdm(range(total_frames), desc="  Analyzing frames", unit="frame"):
        ret, frame = cap.read();
        if not ret: break

        p1_raw = get_health_calibrated(frame, tuple(p1_coords['start']), tuple(p1_coords['end']), lower, upper, p1_max_pixels)
        p2_raw = get_health_calibrated(frame, tuple(p2_coords['start']), tuple(p2_coords['end']), lower, upper, p2_max_pixels)
        
        # (Health filtering and back-fill logic is unchanged)
        if abs(p1_raw - p1_potential) > 0.001: p1_potential, p1_counter = p1_raw, 1
        else: p1_counter += 1
        if p1_counter >= CONFIRMATION_FRAMES and p1_raw < p1_last_confirmed:
            p1_last_confirmed = p1_raw
            for i in range(1, min(len(log_buffer), CONFIRMATION_FRAMES) + 1): log_buffer[-i][1] = custom_round(p1_raw, threshold=0.4)

        if abs(p2_raw - p2_potential) > 0.001: p2_potential, p2_counter = p2_raw, 1
        else: p2_counter += 1
        if p2_counter >= CONFIRMATION_FRAMES and p2_raw < p2_last_confirmed:
            p2_last_confirmed = p2_raw
            for i in range(1, min(len(log_buffer), CONFIRMATION_FRAMES) + 1): log_buffer[-i][2] = custom_round(p2_raw, threshold=0.4)

        FPS, START_TIMER = 30, 40.0
        seconds_passed = frame_count // FPS
        normalized_timer = max(0, START_TIMER - seconds_passed) / START_TIMER

        current_row = [frame_count, custom_round(p1_last_confirmed, threshold=0.5), custom_round(p2_last_confirmed, threshold=0.5), f"{normalized_timer:.3f}"]
        log_buffer.append(current_row)
        
        if len(log_buffer) > CONFIRMATION_FRAMES: final_data.append(log_buffer.pop(0))
    final_data.extend(log_buffer)
    cap.release()
    return final_data

def run_win_rate_analysis(video_path, config):
    """
    NEW: A fast analysis mode that only processes the last frame of a video
    to determine the game's outcome for win-rate calculation.
    """
    print("  > Fast Mode: Analyzing last frame for win rate...")
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 1:
        print("    Warning: Video has no frames. Skipping.")
        cap.release()
        return None
    
    # Seek directly to the last frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("    Warning: Could not read the last frame. Skipping.")
        return None

    # Unpack necessary config values
    p1_coords, p2_coords = config['p1_coords'], config['p2_coords']
    p1_max_pixels, p2_max_pixels = config['p1_calibration']['max_pixels'], config['p2_calibration']['max_pixels']
    lower, upper = np.array(config['color_range']['lower']), np.array(config['color_range']['upper'])

    # Get health values for only the last frame
    p1_health = get_health_calibrated(frame, tuple(p1_coords['start']), tuple(p1_coords['end']), lower, upper, p1_max_pixels)
    p2_health = get_health_calibrated(frame, tuple(p2_coords['start']), tuple(p2_coords['end']), lower, upper, p2_max_pixels)
    
    # The timer value on the last frame isn't critical for win rate, so we can use a placeholder
    # Or calculate it discretely for consistency
    FPS, START_TIMER = 30, 40.0
    seconds_passed = (total_frames - 1) // FPS
    normalized_timer = max(0, START_TIMER - seconds_passed) / START_TIMER

    # Create a CSV-compatible row with the final state
    # Rounding health here is not strictly necessary but good for consistency
    final_row = [total_frames - 1, round(p1_health, 2), round(p2_health, 2), round(normalized_timer, 3)]
    return [final_row] # Return as a list to be compatible with writerows

# ==========================================================
# 4. Main Execution Block
# ==========================================================

def main():
    """
    Main function to orchestrate the analysis of all videos in the input directory.
    Accepts a command-line argument to select the analysis mode.
    """
    parser = argparse.ArgumentParser(description="Tekken Vision Analysis Engine.")
    parser.add_argument(
        '--mode',
        type=str,
        default='full',
        choices=['full', 'win_rate'],
        help="Analysis mode: 'full' for complete data, 'win_rate' for fast outcome-only analysis."
    )
    args = parser.parse_args()

    print(f"--- Tekken Vision Analysis Engine (Mode: {args.mode}) ---")
    config = load_config()
    if config is None: return

    ocr_reader = None
    if args.mode == 'full':
        use_gpu = torch.cuda.is_available()
        print(f"\nInitializing EasyOCR reader (GPU: {use_gpu})...")
        ocr_reader = easyocr.Reader(['en'], gpu=use_gpu)
        print("EasyOCR reader initialized.")

    os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)
    video_files = glob.glob(os.path.join(VIDEO_INPUT_DIR, '*.avi')) + glob.glob(os.path.join(VIDEO_INPUT_DIR, '*.mp4'))
    
    if not video_files:
        print(f"\nWarning: No .avi or .mp4 video files found in '{VIDEO_INPUT_DIR}'.")
        return

    print(f"\nFound {len(video_files)} video(s) to analyze.")
    for video_path in video_files:
        print(f"\n--- Analyzing: {os.path.basename(video_path)} ---")
        
        processed_data = None
        if args.mode == 'full':
            processed_data = run_full_analysis(video_path, config, ocr_reader)
        elif args.mode == 'win_rate':
            processed_data = run_win_rate_analysis(video_path, config)
        
        if processed_data:
            base_filename = os.path.splitext(os.path.basename(video_path))[0]
            output_csv_path = os.path.join(CSV_OUTPUT_DIR, f"{base_filename}.csv")
            header = ['Frame', 'P1_Health', 'P2_Health', 'Normalized_Game_Timer']
            with open(output_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(processed_data)
            print(f"  Success: Analysis complete. Data saved to:\n    '{os.path.relpath(output_csv_path, project_root)}'")
        else:
            print("  Analysis did not produce data. Skipping CSV generation.")

    print("\n--- All Videos Processed ---")

if __name__ == '__main__':
    main()

