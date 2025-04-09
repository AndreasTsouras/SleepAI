import os
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

# ---------------------
# PARAMETERS
# ---------------------
FPS = 25                          # Video frame rate
EPOCH_DURATION = 30               # seconds per epoch
FRAMES_PER_EPOCH = FPS * EPOCH_DURATION  # 25 * 30 = 750 frames per epoch
TOTAL_SAMPLES = 4                 # Total frames sampled per epoch (for example, 2 frames from each half)
FINAL_SIZE = (224, 224)           # Desired resolution

# Transformation: convert PIL image to a tensor and normalize
transform = transforms.Compose([
    transforms.ToTensor(),   # Converts image [0, 255] to tensor [0.0, 1.0]
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
])

# ---------------------
# Helper function: Uniformly sample frames from a list with split strategy
# ---------------------
def sample_epoch_frames_split(frames, total_samples=TOTAL_SAMPLES):
    # This function splits frames into two halves and samples half the samples from each.
    half = len(frames) // 2
    if half < total_samples // 2 or len(frames) - half < total_samples // 2:
        # Not enough frames in one half: fallback to uniform sampling over entire epoch.
        indices = np.linspace(0, len(frames) - 1, total_samples, dtype=int)
        return [frames[i] for i in indices]
    first_half_indices = np.linspace(0, half - 1, total_samples // 2, dtype=int)
    second_half_indices = np.linspace(half, len(frames) - 1, total_samples // 2, dtype=int)
    indices = np.concatenate([first_half_indices, second_half_indices])
    return [frames[i] for i in indices]

# ---------------------
# Process Frames into Tensor Example
# ---------------------
# Assume you have already extracted frames into the folder "extracted_frames" as PNG files.
frames_dir = "extracted_frames"  # Folder with extracted frames from ffmpeg
frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".png")])
print("Total frames extracted:", len(frame_files))

num_epochs = len(frame_files) // FRAMES_PER_EPOCH
print("Number of complete epochs:", num_epochs)

epoch_tensors = []

for epoch in range(num_epochs):
    start_idx = epoch * FRAMES_PER_EPOCH
    end_idx = start_idx + FRAMES_PER_EPOCH
    epoch_files = frame_files[start_idx:end_idx]
    
    # Split the epoch into two halves
    first_half_files = epoch_files[:FRAMES_PER_EPOCH//2]
    second_half_files = epoch_files[FRAMES_PER_EPOCH//2:]
    
    sampled_first = sample_epoch_frames_split(first_half_files, TOTAL_SAMPLES // 2)
    sampled_second = sample_epoch_frames_split(second_half_files, TOTAL_SAMPLES // 2)
    sampled_files = sampled_first + sampled_second  # Total 4 frames per epoch
    frames_list = []
    
    for fname in sampled_files:
        img_path = os.path.join(frames_dir, fname)
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            continue
        tensor_img = transform(img)
        frames_list.append(tensor_img)
    
    # Only add the epoch if exactly TOTAL_SAMPLES frames were sampled.
    if len(frames_list) == TOTAL_SAMPLES:
        # Convert tensor to half precision
        epoch_tensor = torch.stack(frames_list).half()  # shape: [TOTAL_SAMPLES, 3, 224, 224]
        epoch_tensors.append(epoch_tensor)
    else:
        print(f"Epoch {epoch} has {len(frames_list)} frames, expected {TOTAL_SAMPLES}.")

if epoch_tensors:
    # Combine all epochs into one tensor: shape [num_epochs, TOTAL_SAMPLES, 3, 224, 224]
    night_tensor = torch.stack(epoch_tensors)
    output_path = "night_tensor.pt"
    torch.save(night_tensor, output_path)
    print("Saved night tensor with shape:", night_tensor.shape, "to", output_path)
else:
    print("No complete epochs processed.")
