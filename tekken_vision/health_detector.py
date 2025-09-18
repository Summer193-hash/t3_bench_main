import cv2
import numpy as np

def find_bar_edges(frame, rough_start, rough_end, lower_range, upper_range, padding=20):
    """
    Finds the precise pixel start and end of a health bar using color detection.

    This function searches along the horizontal centerline of the provided rough
    coordinates to locate the first and last pixels that match the given HSV color range.

    Args:
        frame (np.array): The video frame to analyze.
        rough_start (tuple): The (x, y) coordinates of the rough start of the bar.
        rough_end (tuple): The (x, y) coordinates of the rough end of the bar.
        lower_range (np.array): The lower bound of the HSV color range.
        upper_range (np.array): The upper bound of the HSV color range.
        padding (int): How many extra pixels to search beyond the rough points.

    Returns:
        tuple: A tuple containing the precise start (x, y) and end (x, y) coordinates.
    """
    x1, y1 = rough_start
    x2, y2 = rough_end
    center_y = (y1 + y2) // 2

    # Define a safe search area based on the rough points and padding
    search_start_x = max(0, min(x1, x2) - padding)
    search_end_x = min(frame.shape[1], max(x1, x2) + padding)
    
    # Extract a single-pixel-high line from the frame for efficient analysis
    line_roi = frame[center_y:center_y + 1, search_start_x:search_end_x]

    hsv_roi = cv2.cvtColor(line_roi, cv2.COLOR_BGR2HSV)
    color_mask = cv2.inRange(hsv_roi, lower_range, upper_range)

    # Find the coordinates of all detected color pixels in the mask
    color_points = np.where(color_mask.flatten() == 255)[0]

    if color_points.size == 0:
        # If no color is found, fall back to the user's original points
        print("Warning: No health bar color found in the search area. Using rough points.")
        return rough_start, rough_end

    # Calculate the precise start and end points
    precise_start_x = search_start_x + color_points.min()
    precise_end_x = search_start_x + color_points.max()

    return (precise_start_x, center_y), (precise_end_x, center_y)

def count_pixels_for_calibration(frame, start_point, end_point, lower_range, upper_range):
    """
    Counts the number of colored pixels in a bar, used for finding the max_pixels
    value when a health bar is at 100%.

    Args:
        frame (np.array): The video frame to analyze.
        start_point (tuple): The precise (x, y) start coordinate of the health bar.
        end_point (tuple): The precise (x, y) end coordinate of the health bar.
        lower_range (np.array): The lower bound of the HSV color range.
        upper_range (np.array): The upper bound of the HSV color range.

    Returns:
        int: The total number of pixels matching the color range.
    """
    x1, y1 = start_point
    x2, _ = end_point
    center_y = y1
    
    health_roi = frame[center_y:center_y + 1, x1:x2 + 1]
    if health_roi.size == 0:
        return 0
        
    hsv_roi = cv2.cvtColor(health_roi, cv2.COLOR_BGR2HSV)
    color_mask = cv2.inRange(hsv_roi, lower_range, upper_range)
    return cv2.countNonZero(color_mask)

def get_health_calibrated(frame, start_point, end_point, lower_range, upper_range, max_pixels):
    """
    Calculates the health percentage based on the number of colored pixels.
    This is the primary function used by the main analysis engine.
    """
    # This function is a placeholder for now. Its full logic will be derived
    # from your main_analyzer.py script.
    current_pixels = count_pixels_for_calibration(frame, start_point, end_point, lower_range, upper_range)
    if max_pixels == 0:
        return 0.0
    return round(current_pixels / max_pixels, 4)

