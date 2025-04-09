import os
import re
import subprocess
import time
from collections import defaultdict

# For tensor building:
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

# ---------------------
# PARAMETERS
# ---------------------
BASE_VIDEO_DIR = r"/mnt/d/Student_Video_project"  # Root folder with patient subfolders.
CROP_FILE = "crop_values.txt"                      # Expected lines, e.g.: "1:bl:250,350, br:550,550, tl:620,75, tr:800,100"
EPOCH_DURATION = 30                               # seconds per epoch
VIDEO_FPS = 25                                    # Video frame rate for tensor creation (e.g., 25 fps)
FRAMES_PER_EPOCH = VIDEO_FPS * EPOCH_DURATION       # e.g., 25 * 30 = 750 frames per epoch
TOTAL_SAMPLES = 4                                 # Total frames sampled per epoch (e.g., 2 from each half)
TARGET_RESOLUTION = "224:224"                     # ffmpeg scale filter target (if used in filter chain)
# NOTE: If you want to extract full frame rate for tensor building, use VIDEO_FPS.
# (In earlier extraction code, extraction_fps was computed by sampling fewer frames;
# here we extract at full FPS so that epochs contain 750 frames.)
extraction_fps = VIDEO_FPS

# Path to your GPU-enabled ffmpeg binary:
ffmpeg_path = "/usr/bin/ffmpeg"  # Replace with your binary's full path

# Output folder where extracted frames will be stored:
output_base_folder = "ffmpeg_extracted_frames"
os.makedirs(output_base_folder, exist_ok=True)

# Option to SAVE intermediate frames.
# (If SAVE_FRAMES is False, output is sent to /dev/null—but then tensor code won’t have files to read.)
SAVE_FRAMES = True

# ---------------------
# TENSOR CREATION PARAMETERS & TRANSFORM
# ---------------------
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts image [0,255] to [0.0,1.0]
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
])

# ---------------------
# STEP 1: Read Crop Values from File
# ---------------------
crop_dict = {}
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
# (Optional: report loaded patient numbers)
print("Loaded crop data for patients:", list(crop_dict.keys()))

# ---------------------
# Helper: Build perspective filter string for a patient.
# ---------------------
def build_perspective_filter(patient_number):
    """
    Build the ffmpeg perspective filter parameter string.
    Maps from source corners to a 224x224 rectangle.
    Expected ffmpeg syntax (8 numeric parameters): tl, tr, bl, br.
    """
    if patient_number not in crop_dict:
        return None
    coords = crop_dict[patient_number]
    # Expected order: top-left, top-right, bottom-right, bottom-left.
    # Our crop file gives: tl, tr, br, bl
    tl = coords['tl']
    tr = coords['tr']
    br = coords['br']
    bl = coords['bl']
    # ffmpeg perspective filter requires eight numeric parameters.
    perspective_params = f"{tl[0]}:{tl[1]}:{tr[0]}:{tr[1]}:{br[0]}:{br[1]}:{bl[0]}:{bl[1]}"
    return perspective_params

# ---------------------
# Helper: Build ffmpeg command for a given video file.
# ---------------------
def build_ffmpeg_command(video_path, out_folder, perspective_filter):
    os.makedirs(out_folder, exist_ok=True)
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    if SAVE_FRAMES:
        output_pattern = os.path.join(out_folder, f"{video_basename}_frame_%05d.png")
    else:
        output_pattern = "/dev/null"
    # Build ffmpeg filter chain.
    # It downloads the frame from GPU (hwdownload), converts format,
    # applies the perspective transform, and extracts at the full frame rate.
    filter_chain = f"hwdownload,format=nv12,perspective={perspective_filter},fps={extraction_fps}"
    command = [
        ffmpeg_path,
        "-hwaccel", "cuda",
        "-hwaccel_output_format", "cuda",
        "-i", video_path,
        "-vf", filter_chain,
        "-vsync", "0",
        output_pattern
    ]
    return command

# ---------------------
# Helper: Process all videos for one patient concurrently.
# ---------------------
def process_videos_for_patient(patient_folder, patient_number):
    # List only .asf files.
    video_files = [f for f in os.listdir(patient_folder) if f.lower().endswith('.asf')]
    if not video_files:
        return

    perspective_filter = build_perspective_filter(patient_number)
    if not perspective_filter:
        return

    # Create an output subfolder for this patient.
    patient_frame_folder = os.path.join(output_base_folder, f"patient_{patient_number}_frames")
    os.makedirs(patient_frame_folder, exist_ok=True)

    # Build ffmpeg commands for every .asf video.
    commands = []
    for video_file in video_files:
        video_path = os.path.join(patient_folder, video_file)
        cmd = build_ffmpeg_command(video_path, patient_frame_folder, perspective_filter)
        commands.append(cmd)

    # Launch all ffmpeg processes concurrently.
    processes = []
    for cmd in commands:
        # (Reduce extraneous output by not printing full command)
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        processes.append(p)
    # Wait for all extraction processes to finish.
    for p in processes:
        stdout, stderr = p.communicate()
        if p.returncode != 0:
            print("ffmpeg error:", stderr)

    # After extraction, process frames into tensors.
    process_video_tensors(patient_frame_folder)

# ---------------------
# Helper: Uniformly sample frames from a list using a split strategy.
# ---------------------
def sample_epoch_frames_split(frames, total_samples=TOTAL_SAMPLES):
    half = len(frames) // 2
    if half < total_samples // 2 or (len(frames) - half) < total_samples // 2:
        # Not enough frames in one half: uniform sampling over the entire epoch.
        indices = np.linspace(0, len(frames) - 1, total_samples, dtype=int)
        return [frames[i] for i in indices]
    first_half_indices = np.linspace(0, half - 1, total_samples // 2, dtype=int)
    second_half_indices = np.linspace(half, len(frames) - 1, total_samples // 2, dtype=int)
    indices = np.concatenate([first_half_indices, second_half_indices])
    return [frames[i] for i in indices]

# ---------------------
# Helper: Process extracted frames for each video into a tensor.
# ---------------------
def process_video_tensor(video_base, frames_list, frames_folder):
    # frames_list: sorted list of frame file names for a single video.
    print(f"Processing video '{video_base}' with {len(frames_list)} extracted frames.")
    num_epochs = len(frames_list) // FRAMES_PER_EPOCH
    epoch_tensors = []
    for epoch in range(num_epochs):
        start_idx = epoch * FRAMES_PER_EPOCH
        end_idx = start_idx + FRAMES_PER_EPOCH
        epoch_frames = frames_list[start_idx:end_idx]
        first_half = epoch_frames[:FRAMES_PER_EPOCH // 2]
        second_half = epoch_frames[FRAMES_PER_EPOCH // 2:]
        sampled_first = sample_epoch_frames_split(first_half, TOTAL_SAMPLES // 2)
        sampled_second = sample_epoch_frames_split(second_half, TOTAL_SAMPLES // 2)
        sampled_files = sampled_first + sampled_second
        tensor_list = []
        for fname in sampled_files:
            img_path = os.path.join(frames_folder, fname)
            try:
                img = Image.open(img_path).convert("RGB")
                tensor_img = transform(img)
                tensor_list.append(tensor_img)
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
        if len(tensor_list) == TOTAL_SAMPLES:
            epoch_tensor = torch.stack(tensor_list).half()
            epoch_tensors.append(epoch_tensor)
        else:
            print(f"Incomplete epoch: expected {TOTAL_SAMPLES} frames, got {len(tensor_list)}")
    if epoch_tensors:
        video_tensor = torch.stack(epoch_tensors)
        output_path = os.path.join(frames_folder, f"{video_base}_tensor.pt")
        torch.save(video_tensor, output_path)
        print(f"Saved tensor for video '{video_base}' with shape {video_tensor.shape} to {output_path}.")
    else:
        print(f"No complete epochs processed for video '{video_base}'.")

# ---------------------
# Helper: Group extracted frames by video base name and process them.
# ---------------------
def process_video_tensors(frames_folder):
    all_frames = [f for f in os.listdir(frames_folder) if f.endswith(".png")]
    groups = defaultdict(list)
    for f in all_frames:
        if "_frame_" in f:
            base = f.split("_frame_")[0]
            groups[base].append(f)
    # Process each video group.
    for video_base, frames in groups.items():
        frames.sort()
        process_video_tensor(video_base, frames, frames_folder)

# ---------------------
# MAIN: Process each patient folder.
# ---------------------
def main():
    # For each patient folder under BASE_VIDEO_DIR.
    for folder_name in os.listdir(BASE_VIDEO_DIR):
        patient_folder = os.path.join(BASE_VIDEO_DIR, folder_name)
        if not os.path.isdir(patient_folder):
            continue
        # Expect folder name like "SDRI001_001_video"; extract patient number as second segment.
        parts = folder_name.split('_')
        if len(parts) < 2:
            continue
        patient_number = parts[1].zfill(3)
        if patient_number not in crop_dict:
            continue
        print(f"\nProcessing patient {patient_number} from folder '{folder_name}'")
        process_videos_for_patient(patient_folder, patient_number)
    print("All processing complete.")

if __name__ == "__main__":
    main()
