import numpy as np
import pandas as pd
import os
import glob
import sys
import cv2
import math

# --- This is the magic snippet ---
# Add the project's root directory to the Python path
# This allows us to import from other project modules if needed in the future
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
# ---------------------------------

# ==========================================================
# 1. Configuration
# ==========================================================
# --- Paths (relative to the project root for portability) ---
NPZ_INPUT_DIR = os.path.join(project_root, 'data', '02_npz_ground_truth')
CSV_OUTPUT_DIR = os.path.join(project_root, 'data', '03_processed_csVs', 'ground_truth_output')
VIDEO_OUTPUT_DIR = os.path.join(project_root, 'data', '03_processed_csVs', 'ground_truth_output', 'videos')

# --- Settings ---
GENERATE_VIDEOS = True  # Set to False to only generate CSVs and skip video creation
CONFIRMATION_FRAMES = 10
VIDEO_FPS = 30
START_TIMER = 40.0
# ==========================================================

def process_single_npz(npz_path):
    """
    Loads a single .npz file, applies all processing logic in-memory,
    and saves the final CSV and an optional video.
    """
    base_filename = os.path.splitext(os.path.basename(npz_path))[0]
    print(f"--- Processing: {base_filename}.npz ---")

    try:
        data = np.load(npz_path, allow_pickle=True)
        images_data = data['images']
        valid_frames_mask = data['valid_frames']
        states_data = data['states']
    except Exception as e:
        print(f"  ERROR: Failed to load or read '{base_filename}.npz'. Skipping. Details: {e}")
        return

    # --- 1. Initial Data Extraction (from read_npz.py logic) ---
    frame_indices = np.where(valid_frames_mask == 1)[0]
    if len(frame_indices) == 0:
        print("  WARNING: No valid frames found in this file. Skipping.")
        return
        
    valid_states = states_data[valid_frames_mask == 1]
    
    # Create an initial DataFrame
    df = pd.DataFrame({
        'Frame': frame_indices,
        'P1_Health': valid_states[:, 0],
        'P2_Health': valid_states[:, 1],
    })

    # --- 2. Apply Buffer & Back-fill Logic (from process_ground_truth.py logic) ---
    p1_last_confirmed = 1.0
    p1_potential = 1.0
    p1_counter = 0
    processed_p1_health = []

    p2_last_confirmed = 1.0
    p2_potential = 1.0
    p2_counter = 0
    processed_p2_health = []

    for index, row in df.iterrows():
        # Player 1 Logic
        if abs(row['P1_Health'] - p1_potential) > 0.001:
            p1_potential = row['P1_Health']
            p1_counter = 1
        else:
            p1_counter += 1

        if p1_counter >= CONFIRMATION_FRAMES and p1_potential < p1_last_confirmed:
            p1_last_confirmed = p1_potential
            # Back-fill the change
            for i in range(min(len(processed_p1_health), CONFIRMATION_FRAMES)):
                processed_p1_health[-(i+1)] = p1_last_confirmed
        
        processed_p1_health.append(p1_last_confirmed)

        # Player 2 Logic
        if abs(row['P2_Health'] - p2_potential) > 0.001:
            p2_potential = row['P2_Health']
            p2_counter = 1
        else:
            p2_counter += 1

        if p2_counter >= CONFIRMATION_FRAMES and p2_potential < p2_last_confirmed:
            p2_last_confirmed = p2_potential
            # Back-fill the change
            for i in range(min(len(processed_p2_health), CONFIRMATION_FRAMES)):
                processed_p2_health[-(i+1)] = p2_last_confirmed
        
        processed_p2_health.append(p2_last_confirmed)

    # Update DataFrame with processed health values
    df['P1_Health'] = processed_p1_health
    df['P2_Health'] = processed_p2_health

    # --- 3. Recalculate and Normalize Timer (from process_ground_truth.py logic) ---
    seconds_passed = df['Frame'] // VIDEO_FPS
    current_timer_value = np.maximum(0, START_TIMER - seconds_passed)
    df['Normalized_Game_Timer'] = current_timer_value / START_TIMER

    # --- 4. Final Formatting and Save CSV ---
    df = df[['Frame', 'P1_Health', 'P2_Health', 'Normalized_Game_Timer']]
    # Apply rounding to 2 decimal places as a final step
    df = df.round(2)
    
    output_csv_path = os.path.join(CSV_OUTPUT_DIR, f"{base_filename}.csv")
    df.to_csv(output_csv_path, index=False, float_format='%.2f')
    print(f"  SUCCESS: Processed CSV saved to data/03.../{base_filename}.csv")

    # --- 5. Optional Video Generation (from read_npz.py logic) ---
    if GENERATE_VIDEOS:
        useful_frames = images_data[valid_frames_mask == 1]
        
        _, frame_height, frame_width = useful_frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_video_path = os.path.join(VIDEO_OUTPUT_DIR, f"{base_filename}.avi")
        out = cv2.VideoWriter(output_video_path, fourcc, VIDEO_FPS, (frame_width, frame_height))

        if not out.isOpened():
            print("  ERROR: Could not initialize VideoWriter for this file.")
            return

        for frame in useful_frames:
            frame_transposed = frame.transpose(1, 2, 0)
            frame_uint8 = frame_transposed.astype(np.uint8)
            frame_ready = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)
            out.write(frame_ready)
        
        out.release()
        print(f"  SUCCESS: Video saved to data/03.../videos/{base_filename}.avi")


def main():
    """Main function to find and process all .npz files in a batch."""
    print("--- Starting Batch Ground Truth Preparation ---")
    
    # Ensure output directories exist
    os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)
    if GENERATE_VIDEOS:
        os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)
    
    npz_files = glob.glob(os.path.join(NPZ_INPUT_DIR, '*.npz'))
    
    if not npz_files:
        print(f"WARNING: No .npz files found in '{NPZ_INPUT_DIR}'. Nothing to do.")
        return
        
    print(f"Found {len(npz_files)} file(s) to process.")
    
    for npz_file_path in npz_files:
        process_single_npz(npz_file_path)
        
    print("\n--- Batch Ground Truth Preparation Complete ---")


if __name__ == "__main__":
    main()


