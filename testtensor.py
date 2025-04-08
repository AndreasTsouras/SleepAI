import cv2
import numpy as np
import os
import re

# File containing the crop values
crop_file = "crop_values.txt"

# Folder containing the first frame images for each patient.
# Assumes files are named like "patient_001_first_frame.jpg" etc.
input_frames_folder = "first_frames"

# Folder to save the warped (transformed) images.
output_folder = "warped_frames"
os.makedirs(output_folder, exist_ok=True)

# Define a regex pattern to parse each line.
# Expected format: "1:bl:250,350, br:550,550, tl:620,75, tr:800,100"
pattern = re.compile(
    r'^(\d+):bl:(\d+),(\d+),\s*br:(\d+),(\d+),\s*tl:(\d+),(\d+),\s*tr:(\d+),(\d+)$'
)

with open(crop_file, "r") as f:
    lines = f.readlines()

for line in lines:
    line = line.strip()
    if not line:
        continue

    match = pattern.match(line)
    if not match:
        print(f"Line did not match expected format: {line}")
        continue

    # Extract values from the regex groups.
    patient_num = match.group(1)
    # Bottom left coordinates:
    bl_x, bl_y = int(match.group(2)), int(match.group(3))
    # Bottom right coordinates:
    br_x, br_y = int(match.group(4)), int(match.group(5))
    # Top left coordinates:
    tl_x, tl_y = int(match.group(6)), int(match.group(7))
    # Top right coordinates:
    tr_x, tr_y = int(match.group(8)), int(match.group(9))

    # For perspective transform, order the source points as:
    # [top-left, top-right, bottom-right, bottom-left]
    src_pts = np.float32([
        [tl_x, tl_y],
        [tr_x, tr_y],
        [br_x, br_y],
        [bl_x, bl_y]
    ])

    # Calculate destination rectangle dimensions using distances:
    width_top = np.linalg.norm(np.array([tr_x, tr_y]) - np.array([tl_x, tl_y]))
    width_bottom = np.linalg.norm(np.array([br_x, br_y]) - np.array([bl_x, bl_y]))
    dest_width = int((width_top + width_bottom) / 2)

    height_left = np.linalg.norm(np.array([bl_x, bl_y]) - np.array([tl_x, tl_y]))
    height_right = np.linalg.norm(np.array([br_x, br_y]) - np.array([tr_x, tr_y]))
    dest_height = int((height_left + height_right) / 2)

    # Define destination points for the transformed image.
    dst_pts = np.float32([
        [0, 0],
        [dest_width, 0],
        [dest_width, dest_height],
        [0, dest_height]
    ])

    # Load the corresponding first frame for this patient.
    input_image_path = os.path.join(input_frames_folder, f"patient_{patient_num}_first_frame.jpg")
    image = cv2.imread(input_image_path)
    if image is None:
        print(f"Could not load image for patient {patient_num} at {input_image_path}")
        continue

    # Compute the perspective transform matrix and warp the image.
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, M, (dest_width, dest_height))

    # Save the warped image so you can verify that it looks correct.
    output_image_path = os.path.join(output_folder, f"warped_patient_{patient_num}.jpg")
    cv2.imwrite(output_image_path, warped)
    print(f"Saved warped image for patient {patient_num} at {os.path.abspath(output_image_path)}")
