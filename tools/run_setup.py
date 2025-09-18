import cv2
import numpy as np
import sys
import os
import json
import glob

# --- Add project root to Python path ---
# This allows us to import from the 'tekken_vision' module
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
# -------------------------------------

from tekken_vision.health_detector import find_bar_edges, count_pixels_for_calibration

# ==========================================================
# 1. Configuration
# ==========================================================
CONFIG_FILE_PATH = os.path.join(project_root, 'config.json')
SAMPLE_VIDEO_DIR = os.path.join(project_root, 'data', '01_raw_videos')
FRAME_FOR_SETUP = 3 # Use a specific frame for consistent setup

# ==========================================================
# 2. Interactive Setup Classes & Functions
# ==========================================================

class HealthBarSetupHelper:
    """A class to hold state during the interactive health bar setup."""
    def __init__(self):
        self.line_points = []
    
    def select_line_points_callback(self, event, x, y, flags, param):
        """Mouse callback to capture the four corner points of the health bars."""
        if event == cv2.EVENT_LBUTTONDOWN and len(self.line_points) < 4:
            self.line_points.append((x, y))
            # Provide visual feedback on the frame
            cv2.circle(param['frame'], (x, y), 5, (0, 255, 0), -1)
            cv2.imshow(param['window_name'], param['frame'])

def get_hsv_from_click(event, x, y, flags, param):
    """Mouse callback to get the average HSV value from a clicked point."""
    if event == cv2.EVENT_LBUTTONDOWN:
        line_roi = param['line_roi']
        clicked_hsv_list = param['clicked_hsv']
        
        # Calculate the original x coordinate on the un-resized ROI
        original_x = int(x * line_roi.shape[1] / param['width'])
        
        # Average a small box around the click for a more stable color reading
        roi_box = line_roi[0:1, max(original_x - 2, 0):min(original_x + 3, line_roi.shape[1])]
        # FIX: Corrected the OpenCV constant from COLOR_BGR_HSV to COLOR_BGR2HSV
        avg_hsv = np.mean(cv2.cvtColor(roi_box, cv2.COLOR_BGR2HSV), axis=(0, 1))
        
        # Store the result
        clicked_hsv_list.clear()
        clicked_hsv_list.append(avg_hsv)
        print(f"  Stored Average HSV: H:{int(avg_hsv[0])}, S:{int(avg_hsv[1])}, V:{int(avg_hsv[2])}")

def interactive_color_calibration(frame, start_point, end_point):
    """
    Opens a window for the user to click on an enlarged health bar segment
    to select its core color and define a color range from it.
    """
    print("\n--- Calibrating Health Bar Color ---")
    print("1. An enlarged image of the health bar will appear.")
    print("2. Click on the most representative part of the health bar color.")
    print("3. After you click, press any key in the window to confirm and continue.")
    
    x1, y1 = start_point
    x2, _ = end_point
    line_roi = frame[y1:y1 + 1, x1:x2 + 1]
    
    window_name = 'Click to Select Health Bar Color'
    cv2.namedWindow(window_name)
    
    # Enlarge the tiny ROI for easier clicking
    resized_width, resized_height = 600, 50
    line_display = cv2.resize(line_roi, (resized_width, resized_height), interpolation=cv2.INTER_NEAREST)
    
    clicked_hsv = []
    callback_params = {'line_roi': line_roi, 'width': resized_width, 'clicked_hsv': clicked_hsv}
    cv2.setMouseCallback(window_name, get_hsv_from_click, callback_params)
    
    while True:
        cv2.imshow(window_name, line_display)
        key = cv2.waitKey(1) & 0xFF
        if key != 255: # Any key press
            if not clicked_hsv:
                print("  Error: Please click on the health bar to select a color before pressing a key.")
            else:
                break
    
    # Define a tolerance range around the selected HSV color
    h, s, v = clicked_hsv[0]
    h_tolerance, s_tolerance, v_tolerance = 5, 50, 50
    lower_range = np.array([max(0, h - h_tolerance), max(0, s - s_tolerance), max(0, v - v_tolerance)])
    upper_range = np.array([min(179, h + h_tolerance), min(255, s + s_tolerance), min(255, v + s_tolerance)])
    
    cv2.destroyAllWindows()
    return lower_range, upper_range

# ==========================================================
# 3. Main Setup Workflow
# ==========================================================

def run_health_bar_setup(frame):
    """Orchestrates the full health bar setup process: coordinates, color, and pixels."""
    setup_helper = HealthBarSetupHelper()
    window_name = 'Setup: Click 4 health bar points (ESC to cancel)'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, setup_helper.select_line_points_callback, {'frame': frame, 'window_name': window_name})
    
    instructions = ["1. P1 Left Edge", "2. P1 Right Edge", "3. P2 Left Edge", "4. P2 Right Edge"]
    print("\n--- Step 2: Health Bar Coordinate Setup ---")
    
    while len(setup_helper.line_points) < 4:
        frame_copy = frame.copy()
        instruction_text = instructions[len(setup_helper.line_points)]
        
        # Add clear on-screen instructions
        cv2.putText(frame_copy, instruction_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)
        cv2.putText(frame_copy, instruction_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow(window_name, frame_copy)
        
        if cv2.waitKey(20) & 0xFF == 27: # ESC key
            cv2.destroyAllWindows()
            return None
            
    cv2.destroyAllWindows()

    print("--- Step 3: Health Bar Color & Pixel Calibration ---")
    # Use a broad, temporary color range to find the edges initially
    temp_lower = np.array([80, 80, 100])
    temp_upper = np.array([100, 255, 255])
    
    p1_rough_start, p1_rough_end = setup_helper.line_points[0], setup_helper.line_points[1]
    p1_start, p1_end = find_bar_edges(frame, p1_rough_start, p1_rough_end, temp_lower, temp_upper)
    
    # Get the precise color range from user input, assuming both bars are the same color
    lower_range, upper_range = interactive_color_calibration(frame, p1_start, p1_end)
    
    print("  Refining bar edges with calibrated color values...")
    p1_start, p1_end = find_bar_edges(frame, p1_rough_start, p1_rough_end, lower_range, upper_range)
    p2_rough_start, p2_rough_end = setup_helper.line_points[2], setup_helper.line_points[3]
    p2_start, p2_end = find_bar_edges(frame, p2_rough_start, p2_rough_end, lower_range, upper_range)

    # Count the pixels at full health for normalization
    # This assumes the setup frame shows both players at full health
    p1_max_pixels = count_pixels_for_calibration(frame, p1_start, p1_end, lower_range, upper_range)
    p2_max_pixels = count_pixels_for_calibration(frame, p2_start, p2_end, lower_range, upper_range)

    return p1_start, p1_end, p2_start, p2_end, lower_range, upper_range, p1_max_pixels, p2_max_pixels

def run_timer_roi_setup(frame):
    """Allows the user to select the timer's region of interest (ROI)."""
    print("\n--- Step 4: Timer ROI Setup ---")
    print("1. A window will appear.")
    print("2. Use your mouse to click and drag a box tightly around the game timer.")
    print("3. Press ENTER or SPACE to confirm.")
    print("4. Press 'C' or ESC to cancel the selection.")
    
    roi = cv2.selectROI("Select Timer ROI", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()
    
    if roi == (0, 0, 0, 0): # Check if selection was cancelled
        print("  Timer selection cancelled.")
        return None
    return roi

# ==========================================================
# 4. Main Execution Block
# ==========================================================

def main():
    """Main function to orchestrate the entire setup process."""
    print("--- Tekken Vision Analyzer Setup Utility ---")
    
    # --- Video Selection ---
    video_files = glob.glob(os.path.join(SAMPLE_VIDEO_DIR, '*.avi')) + glob.glob(os.path.join(SAMPLE_VIDEO_DIR, '*.mp4'))
    if not video_files:
        print(f"Error: No .avi or .mp4 video files found in '{SAMPLE_VIDEO_DIR}'.")
        print("Please add a sample video to that directory to run the setup.")
        return
        
    print("\nPlease select a video to use for calibration:")
    for i, video in enumerate(video_files):
        print(f"  [{i+1}] {os.path.basename(video)}")
        
    choice = -1
    while choice < 1 or choice > len(video_files):
        try:
            choice = int(input(f"Enter number (1-{len(video_files)}): "))
        except ValueError:
            pass
            
    selected_video_path = video_files[choice-1]
    print(f"Using '{os.path.basename(selected_video_path)}' for setup.")

    # --- Frame Selection ---
    cap = cv2.VideoCapture(selected_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at '{selected_video_path}'")
        return
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame_pos = FRAME_FOR_SETUP
    
    print("\n--- Step 1: Frame Selection ---")
    print("Use 'd' to go forward, 'a' to go back.")
    print("Press ENTER to select the current frame for calibration.")
    print("Press ESC to quit.")
    
    selected_frame = None
    
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_pos)
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame {current_frame_pos}. Stepping back.")
            current_frame_pos = max(0, current_frame_pos - 1)
            continue
            
        frame_copy = frame.copy()
        display_text = f"Frame: {current_frame_pos}/{total_frames - 1} | A/D = Change | ENTER = Select | ESC = Quit"
        cv2.putText(frame_copy, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(frame_copy, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        cv2.imshow("Select Calibration Frame", frame_copy)
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('d'): # 'd' key to move forward
            current_frame_pos = min(total_frames - 1, current_frame_pos + 1)
        elif key == ord('a'): # 'a' key to move back
            current_frame_pos = max(0, current_frame_pos - 1)
        elif key == 13: # Enter key to confirm
            selected_frame = frame
            cv2.destroyAllWindows()
            break
        elif key == 27: # ESC key to quit
            cv2.destroyAllWindows()
            break
            
    cap.release()

    if selected_frame is None:
        print("Frame selection cancelled. Aborting setup.")
        return
        
    print(f"Proceeding with calibration using frame {current_frame_pos}.")
    
    # --- Run Setup Processes ---
    health_results = run_health_bar_setup(selected_frame.copy())
    if not health_results:
        print("\nHealth bar setup was cancelled. Aborting.")
        return
        
    timer_roi = run_timer_roi_setup(selected_frame.copy())
    if not timer_roi:
        print("\nTimer setup was cancelled. Aborting.")
        return

    # --- Assemble and Save Configuration ---
    p1_start, p1_end, p2_start, p2_end, lower, upper, p1_pixels, p2_pixels = health_results
    
    # FIX: Explicitly cast all NumPy numeric types to standard Python types for JSON compatibility.
    config_data = {
        "p1_coords": {"start": (int(p1_start[0]), int(p1_start[1])), "end": (int(p1_end[0]), int(p1_end[1]))},
        "p2_coords": {"start": (int(p2_start[0]), int(p2_start[1])), "end": (int(p2_end[0]), int(p2_end[1]))},
        "p1_calibration": {"max_pixels": int(p1_pixels)},
        "p2_calibration": {"max_pixels": int(p2_pixels)},
        "color_range": {"lower": [int(c) for c in lower], "upper": [int(c) for c in upper]},
        "timer_roi": [int(c) for c in timer_roi] # Convert tuple to a list of standard ints
    }
    
    try:
        with open(CONFIG_FILE_PATH, 'w') as f:
            json.dump(config_data, f, indent=4)
        print("\n--------------------------------------------------")
        print("Success: Configuration saved to 'config.json'")
        print("You can now run the main analysis script.")
        print("--------------------------------------------------")
    except Exception as e:
        print(f"\nError: Could not save configuration file. Details: {e}")

if __name__ == '__main__':
    main()

