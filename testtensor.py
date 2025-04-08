import os
import re
import cv2
import numpy as np

# --- Step 1: Read Crop Values from File ---

# File containing crop values in the format:
# "1:bl:250,350, br:550,550, tl:620,75, tr:800,100"
crop_file = "crop_values.txt"

crop_dict = {}
pattern = re.compile(
    r'^(\d+):bl:(\d+),(\d+),\s*br:(\d+),(\d+),\s*tl:(\d+),(\d+),\s*tr:(\d+),(\d+)$'
)

with open(crop_file, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        match = pattern.match(line)
        if match:
            # Pad patient number to 3 digits (e.g., "1" becomes "001")
            patient_number = match.group(1).zfill(3)
            crop_dict[patient_number] = {
                'bl': (int(match.group(2)), int(match.group(3))),
                'br': (int(match.group(4)), int(match.group(5))),
                'tl': (int(match.group(6)), int(match.group(7))),
                'tr': (int(match.group(8)), int(match.group(9)))
            }
        else:
            print(f"Line did not match expected format: {line}")

# Debug: Print out the crop dictionary
print("Crop data loaded for patients:")
for pnum, coords in crop_dict.items():
    print(f" Patient {pnum}: {coords}")

# --- Step 2: Process First Frame Images ---

input_frames_folder = "first_frames"  # Folder containing first frame images
output_folder = "warped_frames"
os.makedirs(output_folder, exist_ok=True)

# Debug: Print current working directory
import os
print("Current working directory:", os.getcwd())

# Iterate over each file in the first_frames folder
files_processed = 0
for filename in os.listdir(input_frames_folder):
    # Debug: print each filename encountered
    print(f"Found file: {filename}")

    # Expecting filenames like "patient_001_first_frame.jpg"
    if not filename.lower().endswith('_first_frame.jpg'):
        print(f"Skipping {filename}; does not match expected pattern.")
        continue

    m = re.search(r'patient_(\d+)_first_frame', filename)
    if not m:
        print(f"Filename '{filename}' does not match expected pattern; skipping.")
        continue

    patient_num = m.group(1).zfill(3)
    print(f"Processing file for patient {patient_num}")

    if patient_num not in crop_dict:
        print(f"No crop data found for patient {patient_num}; skipping {filename}.")
        continue

    coords = crop_dict[patient_num]
    # Order the source points: top-left, top-right, bottom-right, bottom-left.
    src_pts = np.float32([
        list(coords['tl']),
        list(coords['tr']),
        list(coords['br']),
        list(coords['bl'])
    ])

    # Compute destination dimensions based on average widths and heights.
    width_top = np.linalg.norm(np.array(coords['tr']) - np.array(coords['tl']))
    width_bottom = np.linalg.norm(np.array(coords['br']) - np.array(coords['bl']))
    dest_width = int((width_top + width_bottom) / 2)

    height_left = np.linalg.norm(np.array(coords['bl']) - np.array(coords['tl']))
    height_right = np.linalg.norm(np.array(coords['br']) - np.array(coords['tr']))
    dest_height = int((height_left + height_right) / 2)

    dst_pts = np.float32([
        [0, 0],
        [dest_width, 0],
        [dest_width, dest_height],
        [0, dest_height]
    ])

    # Load the image.
    image_path = os.path.join(input_frames_folder, filename)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        continue

    # Compute perspective transform and warp the image.
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, M, (dest_width, dest_height))

    # Save the warped image.
    output_path = os.path.join(output_folder, f"warped_patient_{patient_num}.jpg")
    success = cv2.imwrite(output_path, warped)
    if success:
        print(f"Saved warped image for patient {patient_num} at {os.path.abspath(output_path)}")
        files_processed += 1
    else:
        print(f"Failed to save warped image for patient {patient_num}")

print(f"Processing complete. {files_processed} file(s) processed and saved in '{output_folder}'")
