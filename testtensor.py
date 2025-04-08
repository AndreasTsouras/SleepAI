import cv2
import numpy as np
import re
import os

# Path to the file that contains the crop values.
crop_file = "crop_values.txt"  # Update this path as needed

# Read the crop values from the file.
with open(crop_file, "r") as f:
    line = f.readline().strip()
    # Example line: "1: bl250,350, br:550,550, tl:620,75, tr:800,100"
    # Extract the patient number (if needed) and the rest of the coordinate string.
    patient_num, coords_str = line.split(":", 1)
    patient_num = patient_num.strip()
    
    # Use regex to match patterns: key (bl, br, tl, tr) followed optionally by a colon, then two numbers.
    pattern = r"(bl|br|tl|tr):?(\d+),(\d+)"
    matches = re.findall(pattern, coords_str)
    
    # Build a dictionary of coordinates.
    coords = {}
    for key, x, y in matches:
        coords[key] = (int(x), int(y))

print("Parsed crop coordinates:", coords)

# Ensure that all expected keys exist.
required_keys = ['tl', 'tr', 'br', 'bl']
for key in required_keys:
    if key not in coords:
        print(f"Error: missing coordinate for {key} in file.")
        exit()

# Define the source points for the perspective transform.
# Order required: top-left, top-right, bottom-right, bottom-left.
src_pts = np.float32([coords['tl'], coords['tr'], coords['br'], coords['bl']])
print("Source points:", src_pts)

# Define a destination rectangle.
# For testing, we use fixed dimensions (adjust these if needed).
dst_width = 361
dst_height = 515
dst_pts = np.float32([[0, 0],
                      [dst_width, 0],
                      [dst_width, dst_height],
                      [0, dst_height]])
print("Destination points:", dst_pts)

# Load a test image (e.g., a saved first frame for the patient).
input_image = "first_frame.jpg"  # Update with the actual path if different
image = cv2.imread(input_image)
if image is None:
    print("Error: Test image not found!")
    exit()

# Compute the perspective transform matrix.
M = cv2.getPerspectiveTransform(src_pts, dst_pts)

# Apply the perspective warp.
warped = cv2.warpPerspective(image, M, (dst_width, dst_height))

# Save the warped image so you can visually check the output.
output_image = "warped_from_file.jpg"
cv2.imwrite(output_image, warped)
print("Warped image saved to:", os.path.abspath(output_image))
