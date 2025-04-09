import os
import subprocess
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

# ===============================
# Configuration
# ===============================
video_file = "input.asf"   # Path to your video file
output_frames_folder = "extracted_frames"  # Folder where frames will be saved
os.makedirs(output_frames_folder, exist_ok=True)

# Full path to your GPU-enabled ffmpeg binary
# Adjust the path accordingly (e.g. if it's in your project folder)
ffmpeg_path = "/mnt/d/Andrew-SleepAI/ffmpeg_gpu/ffmpeg"  

# We want frames scaled to 224x224 (final resolution expected by our model)
target_resolution = "224:224"

# For a 30-second epoch at 25 fps, there are 750 frames.
# To uniformly sample 16 frames per epoch, we need 16 frames over 30 seconds,
# which means one frame every (30 / 16) ≈ 1.875 seconds.
# Thus, we set the extraction fps to 1 / 1.875 ≈ 0.5333.
extraction_fps = 1 / 1.875  # ~0.5333 fps

# ===============================
# Step 1: Extract Frames Using ffmpeg (GPU-accelerated)
# ===============================
# The command uses:
#   -hwaccel cuda and -hwaccel_output_format cuda: Use GPU for decoding.
#   -vf "fps=0.5333,scale=224:224": Extract frames at ~0.5333 fps and resize to 224x224.
#   -vsync 0: ensures no frames are dropped or duplicated.
output_pattern = os.path.join(output_frames_folder, "frame_%05d.png")

ffmpeg_command = [
    ffmpeg_path,
    "-hwaccel", "cuda",
    "-hwaccel_output_format", "cuda",
    "-i", video_file,
    "-vf", f"fps={extraction_fps},scale={target_resolution}",
    "-vsync", "0",
    output_pattern
]

print("Running ffmpeg command:")
print(" ".join(ffmpeg_command))
try:
    subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)
    print("ffmpeg completed successfully.")
except subprocess.CalledProcessError as e:
    print("ffmpeg encountered an error:")
    print(e.stderr)
    exit(1)

# ===============================
# Step 2: Process Extracted Frames into Epoch Tensors
# ===============================
# At 0.5333 fps, every 30 seconds should yield roughly 16 frames.
# List all extracted frame files (ensure they are sorted in order).
frame_files = sorted([f for f in os.listdir(output_frames_folder) if f.endswith(".png")])
total_frames = len(frame_files)
print("Total frames extracted:", total_frames)

# Calculate number of complete epochs (each epoch should have exactly 16 frames)
frames_per_epoch = 16
num_epochs = total_frames // frames_per_epoch
print("Number of complete epochs:", num_epochs)

# Define transformation (convert PIL image to tensor and normalize)
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts image pixel values to [0.0, 1.0]
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
])

epoch_tensors = []
for epoch in range(num_epochs):
    # Get filenames for the current epoch.
    start_idx = epoch * frames_per_epoch
    end_idx = start_idx + frames_per_epoch
    epoch_files = frame_files[start_idx:end_idx]
    
    frames_list = []
    for fname in epoch_files:
        img_path = os.path.join(output_frames_folder, fname)
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            continue
        tensor_img = transform(img)
        frames_list.append(tensor_img)
    
    if len(frames_list) == frames_per_epoch:
        epoch_tensor = torch.stack(frames_list)  # shape: [16, 3, 224, 224]
        epoch_tensors.append(epoch_tensor)
    else:
        print(f"Epoch {epoch} has {len(frames_list)} frames, expected {frames_per_epoch}.")

if epoch_tensors:
    # Combine all epochs into a single tensor.
    # Final shape: [num_epochs, 16, 3, 224, 224]
    night_tensor = torch.stack(epoch_tensors)
    output_tensor_path = "night_tensor.pt"
    torch.save(night_tensor, output_tensor_path)
    print("Saved night tensor with shape:", night_tensor.shape, "to", output_tensor_path)
else:
    print("No complete epochs processed.")
