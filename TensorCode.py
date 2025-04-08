import os
import re
import cv2
import numpy as np
import torch
from torchvision import transforms

# ---------------------
# PARAMETERS
# ---------------------
BASE_VIDEO_DIR = r"/mnt/d/Student_Video_project"  # Root directory containing patient folders.
CROP_FILE = "crop_values.txt"  # Contains lines like: 1:bl:250,350, br:550,550, tl:620,75, tr:800,100
EPOCH_DURATION = 30         # seconds per epoch
SAMPLE_COUNT = 16           # Number of frames to sample per epoch (adjust as needed)
FINAL_SIZE = (224, 224)     # Final resolution (width, height)

# Transformation to convert warped images to tensor (and normalize if desired).
# You can adjust mean and std if needed.
transform = transforms.Compose([
    transforms.ToTensor(),   # converts image (H x W x C) [0,255] to tensor with [0.0,1.0]
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
])

# ---------------------
# STEP 1: Read Crop Values from File
# ---------------------
crop_dict = {}  # keys are patient numbers as 3-digit strings, e.g. "001"
crop_pattern = re.compile(
    r'^(\d+):bl:(\d+),(\d+),\s*br:(\d+),(\d+),\s*tl:(\d+),(\d+),\s*tr:(\d+),(\d+)$'
)

with open(CROP_FILE, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        match = crop_pattern.match(line)
        if match:
            patient_number = match.group(1).zfill(3)
            crop_dict[patient_number] = {
                'bl': (int(match.group(2)), int(match.group(3))),
                'br': (int(match.group(4)), int(match.group(5))),
                'tl': (int(match.group(6)), int(match.group(7))),
                'tr': (int(match.group(8)), int(match.group(9)))
            }
        else:
            print(f"Line did not match expected format: {line}")

print("Loaded crop data for patients:", list(crop_dict.keys()))

# ---------------------
# STEP 2: Process Each Patient Folder
# ---------------------

# We will save the per-patient tensor in an output directory.
tensor_output_folder = "patient_night_tensors"
os.makedirs(tensor_output_folder, exist_ok=True)

# Iterate through each patient folder in the base directory.
for folder_name in os.listdir(BASE_VIDEO_DIR):
    patient_folder = os.path.join(BASE_VIDEO_DIR, folder_name)
    if not os.path.isdir(patient_folder):
        continue

    # Assume the folder naming is like "SDRI001_001_video" and patient number is the second part.
    parts = folder_name.split('_')
    if len(parts) < 2:
        print(f"Folder '{folder_name}' does not match expected pattern; skipping.")
        continue

    patient_number = parts[1].zfill(3)
    print(f"\nProcessing patient {patient_number} in folder '{folder_name}'")
    
    if patient_number not in crop_dict:
        print(f"No crop data for patient {patient_number}; skipping folder.")
        continue

    # Compute the perspective transformation matrix for this patient based on crop values.
    coords = crop_dict[patient_number]
    # Source points in the order: top-left, top-right, bottom-right, bottom-left.
    src_pts = np.float32([
        list(coords['tl']),
        list(coords['tr']),
        list(coords['br']),
        list(coords['bl'])
    ])
    # Compute an approximate destination rectangle using average distances.
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
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    # For collecting epoch tensors for the patient.
    patient_epoch_tensors = []

    # Now iterate through each video file in the patient folder. Assuming .asf extension.
    video_files = [f for f in os.listdir(patient_folder) if f.lower().endswith('.asf')]
    video_files.sort()  # sort files in order (if needed)
    
    if not video_files:
        print(f"No video files found in {folder_name}.")
        continue

    # Process each video file.
    for video_file in video_files:
        video_path = os.path.join(patient_folder, video_file)
        print(f" Processing video: {video_file}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  Could not open video file {video_file}.")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            print(f"  Invalid FPS for {video_file}; skipping.")
            cap.release()
            continue
        frames_per_epoch = int(fps * EPOCH_DURATION)
        epoch_frames = []
        frame_index = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                # End of video: if any frames remain that form a partial epoch, we may process them too.
                if len(epoch_frames) >= SAMPLE_COUNT:
                    # Uniformly sample from the final partial epoch.
                    indices = np.linspace(0, len(epoch_frames) - 1, SAMPLE_COUNT, dtype=int)
                    sampled_frames = [epoch_frames[i] for i in indices]
                    epoch_tensor = torch.stack(sampled_frames)  # shape: [SAMPLE_COUNT, 3, 224, 224]
                    patient_epoch_tensors.append(epoch_tensor)
                break

            frame_index += 1

            # Apply the perspective transform to the frame.
            warped = cv2.warpPerspective(frame, M, (dest_width, dest_height))
            # Resize the warped image to the desired final size (224x224).
            warped_resized = cv2.resize(warped, FINAL_SIZE)
            # Convert to tensor and normalize.
            tensor_img = transform(cv2.cvtColor(warped_resized, cv2.COLOR_BGR2RGB))
            epoch_frames.append(tensor_img)

            # Once we have a full epoch of frames, sample SAMPLE_COUNT frames uniformly.
            if frame_index % frames_per_epoch == 0:
                if len(epoch_frames) >= SAMPLE_COUNT:
                    indices = np.linspace(0, len(epoch_frames) - 1, SAMPLE_COUNT, dtype=int)
                    sampled_frames = [epoch_frames[i] for i in indices]
                    epoch_tensor = torch.stack(sampled_frames)  # shape: [SAMPLE_COUNT, 3, 224, 224]
                    patient_epoch_tensors.append(epoch_tensor)
                epoch_frames = []  # reset for next epoch

        cap.release()

    if patient_epoch_tensors:
        # Combine all epochs into a single tensor of shape [num_epochs, SAMPLE_COUNT, 3, 224, 224].
        night_tensor = torch.stack(patient_epoch_tensors)
        output_filename = f"patient_{patient_number}_night_tensor.pt"
        output_path = os.path.join(tensor_output_folder, output_filename)
        torch.save(night_tensor, output_path)
        print(f"Saved tensor for patient {patient_number} with shape {night_tensor.shape} to {output_path}")
    else:
        print(f"No epochs processed for patient {patient_number}.")

print("Processing complete.")
