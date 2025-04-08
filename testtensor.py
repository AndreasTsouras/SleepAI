import os
import re
import cv2
import numpy as np

# --- Step 1: Read Crop Values from File (using line order) ---
crop_file = "crop_values.txt"  # Each row represents one patient in order.

crop_dict = {}
# Regex pattern to extract coordinates in the format:
# "001:bl:250,350, br:550,550, tl:620,75, tr:800,100"
pattern = re.compile(
    r'^(?:\d+):bl:(\d+),(\d+),\s*br:(\d+),(\d+),\s*tl:(\d+),(\d+),\s*tr:(\d+),(\d+)$'
)

with open(crop_file, "r") as f:
    lines = f.readlines()

for idx, line in enumerate(lines):
    line = line.strip()
    if not line:
        continue
    match = pattern.match(line)
    if match:
        # Override the patient number with the line order:
        patient_number = str(idx + 1).zfill(3)  # e.g., 1 becomes "001"
        crop_dict[patient_number] = {
            'bl': (int(match.group(1)), int(match.group(2))),
            'br': (int(match.group(3)), int(match.group(4))),
            'tl': (int(match.group(5)), int(match.group(6))),
            'tr': (int(match.group(7)), int(match.group(8)))
        }
    else:
        print(f"Line did not match expected format: {line}")

# Debug: print loaded crop keys to verify order
print("Loaded crop data keys:", list(crop_dict.keys()))

# --- Step 2: Process First Frame Images Using Crop Data ---
input_frames_folder = "first_frames"  # Folder with first frame images, e.g. "patient_001_first_frame.jpg"
output_folder = "warped_frames"
os.makedirs(output_folder, exist_ok=True)

print("Current working directory:", os.getcwd())

files_processed = 0
for filename in os.listdir(input_frames_folder):
    if not filename.lower().endswith('_first_frame.jpg'):
        continue

    # Extract patient number from filename; expecting format "patient_XXX_first_frame.jpg"
    import re
    m = re.search(r'patient_(\d+)_first_frame', filename)
    if not m:
        print(f"Filename '{filename}' does not match expected pattern; skipping.")
        continue

    patient_num = m.group(1)
    print(f"Processing file for patient {patient_num} (from filename {filename})")

    if patient_num not in crop_dict:
        print(f"No crop data found for patient {patient_num}. Available keys: {list(crop_dict.keys())}")
        continue

    coords = crop_dict[patient_num]
    # For the perspective transform, order points as: top-left, top-right, bottom-right, bottom-left.
    src_pts = np.float32([
        list(coords['tl']),
        list(coords['tr']),
        list(coords['br']),
        list(coords['bl'])
    ])

    # Compute average destination dimensions based on distances:
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

    image_path = os.path.join(input_frames_folder, filename)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        continue

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, M, (dest_width, dest_height))

    output_path = os.path.join(output_folder, f"warped_patient_{patient_num}.jpg")
    success = cv2.imwrite(output_path, warped)
    if success:
        print(f"Saved warped image for patient {patient_num} to {os.path.abspath(output_path)}")
        files_processed += 1
    else:
        print(f"Failed to save warped image for patient {patient_num}")

print(f"Processing complete. {files_processed} file(s) processed and saved in '{output_folder}'.")
