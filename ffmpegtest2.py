import os
import re
import cv2
import numpy as np
import torch
from torchvision import transforms
import concurrent.futures

# ---------------------
# PARAMETERS
# ---------------------
BASE_VIDEO_DIR = r"/mnt/d/Student_Video_project"  # Root directory containing patient folders.
CROP_FILE = "crop_values.txt"  # Crop file with lines like: 1:bl:250,350, br:550,550, tl:620,75, tr:800,100
EPOCH_DURATION = 30         # seconds per epoch
SAMPLE_COUNT = 16           # Total sample frames per epoch
FINAL_SIZE = (224, 224)     # Final resolution (width, height)
FPS = 25                    # Video frame rate

# Transformation: convert a PIL image to tensor and normalize.
transform = transforms.Compose([
    transforms.ToTensor(),   # Converts image pixels [0, 255] to [0.0, 1.0]
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
])

# ---------------------
# STEP 1: Read Crop Values from File
# ---------------------
crop_dict = {}  # keys: patient number as a 3-digit string (e.g., "001")
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
# Helper: Process a Single Video File
# ---------------------
def process_video_file(video_path, M, dest_dims, frames_per_epoch, FINAL_SIZE, transform, SAMPLE_COUNT):
    """
    Process a single video file:
      - Instead of processing every frame, calculate sample indices for each epoch.
      - For each epoch, only process frames whose index is in the computed sample indices.
      - Apply perspective transform, resize, and transform.
      - Return a list of epoch tensors (each of shape [SAMPLE_COUNT, 3, 224, 224]).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video file: {video_path}")
        return []
    
    video_epoch_tensors = []
    epoch_samples = []
    frame_index = 0
    # Compute sample indices once per epoch.
    sample_indices = np.linspace(0, frames_per_epoch - 1, SAMPLE_COUNT, dtype=int)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            # If there are leftover frames in the current epoch, and we have exactly SAMPLE_COUNT processed
            if len(epoch_samples) == SAMPLE_COUNT:
                video_epoch_tensors.append(torch.stack(epoch_samples))
            break
        
        # If the current frame in the epoch is one we want to sample
        if frame_index in sample_indices:
            # Apply perspective transform using the given matrix M.
            warped = cv2.warpPerspective(frame, M, dest_dims)
            # Resize to FINAL_SIZE (224x224)
            warped_resized = cv2.resize(warped, FINAL_SIZE)
            # Convert BGR to RGB and then to tensor.
            tensor_img = transform(cv2.cvtColor(warped_resized, cv2.COLOR_BGR2RGB))
            epoch_samples.append(tensor_img)
        
        frame_index += 1
        
        # Check if we've completed an epoch.
        if frame_index == frames_per_epoch:
            if len(epoch_samples) == SAMPLE_COUNT:
                video_epoch_tensors.append(torch.stack(epoch_samples))  # shape: [SAMPLE_COUNT, 3, 224, 224]
            else:
                print(f"Warning: Epoch in {video_path} had {len(epoch_samples)} frames (expected {SAMPLE_COUNT}).")
            # Reset for the next epoch.
            frame_index = 0
            epoch_samples = []
            sample_indices = np.linspace(0, frames_per_epoch - 1, SAMPLE_COUNT, dtype=int)
    
    cap.release()
    return video_epoch_tensors

# ---------------------
# STEP 2: Process Each Patient Folder Concurrently
# ---------------------
tensor_output_folder = "patient_night_tensors"
os.makedirs(tensor_output_folder, exist_ok=True)

# For each patient folder in the base directory.
for folder_name in os.listdir(BASE_VIDEO_DIR):
    patient_folder = os.path.join(BASE_VIDEO_DIR, folder_name)
    if not os.path.isdir(patient_folder):
        continue

    # Expect folder names like "SDRI001_001_video"; patient number is the second segment.
    parts = folder_name.split('_')
    if len(parts) < 2:
        print(f"Folder '{folder_name}' does not match expected pattern; skipping.")
        continue

    patient_number = parts[1].zfill(3)
    print(f"\nProcessing patient {patient_number} from folder '{folder_name}'")
    
    if patient_number not in crop_dict:
        print(f"No crop data for patient {patient_number}; skipping folder.")
        continue

    # Compute the perspective transformation matrix M for this patient.
    coords = crop_dict[patient_number]
    # Order source points as: top-left, top-right, bottom-right, bottom-left.
    src_pts = np.float32([
        list(coords['tl']),
        list(coords['tr']),
        list(coords['br']),
        list(coords['bl'])
    ])
    # Compute destination dimensions from crop coordinates.
    width_top = np.linalg.norm(np.array(coords['tr']) - np.array(coords['tl']))
    width_bottom = np.linalg.norm(np.array(coords['br']) - np.array(coords['bl']))
    dest_width = int((width_top + width_bottom) / 2)
    height_left = np.linalg.norm(np.array(coords['bl']) - np.array(coords['tl']))
    height_right = np.linalg.norm(np.array(coords['br']) - np.array(coords['tr']))
    dest_height = int((height_left + height_right) / 2)
    dest_dims = (dest_width, dest_height)
    M = cv2.getPerspectiveTransform(src_pts, dest_dims)
    
    frames_per_epoch = int(FPS * EPOCH_DURATION)  # e.g. 25*30=750
    
    # Gather video files in the patient folder.
    video_files = [f for f in os.listdir(patient_folder) if f.lower().endswith('.asf')]
    video_files.sort()
    if not video_files:
        print(f"No video files found in {folder_name}.")
        continue

    patient_epoch_tensors = []
    # We'll run up to three concurrent processes.
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
        futures = []
        for video_file in video_files:
            video_path = os.path.join(patient_folder, video_file)
            futures.append(executor.submit(process_video_file, video_path, M, dest_dims, frames_per_epoch, FINAL_SIZE, transform, SAMPLE_COUNT))
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                patient_epoch_tensors.extend(result)
    
    if patient_epoch_tensors:
        # Combine all epoch tensors into one tensor with shape [num_epochs, SAMPLE_COUNT, 3, 224, 224].
        night_tensor = torch.stack(patient_epoch_tensors)
        output_filename = f"patient_{patient_number}_night_tensor.pt"
        output_path = os.path.join(tensor_output_folder, output_filename)
        torch.save(night_tensor, output_path)
        print(f"Saved tensor for patient {patient_number} with shape {night_tensor.shape} to {output_path}")
    else:
        print(f"No epochs processed for patient {patient_number}.")

print("Processing complete.")
