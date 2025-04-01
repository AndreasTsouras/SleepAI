import os
import subprocess
import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# ---------------------------
# Step 1: GPU-accelerated ffmpeg Conversion
# ---------------------------

# Set your input folder containing the .asf files (Windows drive, accessed via WSL)
input_folder = "/mnt/e/SDRI001_001_video"  # update as needed

# Create a folder to store the pre-processed videos (cropped & scaled)
preprocessed_folder = "preprocessed_videos"
os.makedirs(preprocessed_folder, exist_ok=True)

# Define the cropping parameters (example values)
bed_x, bed_y, bed_w, bed_h = 100, 200, 300, 200

# Construct the ffmpeg video filter to crop to the bed region and scale to 224x224.
# Format: crop=width:height:x:y, then scale.
crop_filter = f"crop={bed_w}:{bed_h}:{bed_x}:{bed_y},scale=224:224"

# List all .asf files in the input folder and sort them by the number after the underscore.
video_files = [f for f in os.listdir(input_folder) if f.endswith('.asf')]
video_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

print("Converting raw videos to preprocessed MP4s with GPU acceleration...")
for video_file in tqdm(video_files, desc="Converting videos"):
    input_path = os.path.join(input_folder, video_file)
    output_file = video_file.replace(".asf", ".mp4")
    output_path = os.path.join(preprocessed_folder, output_file)
    ffmpeg_cmd = [
         "ffmpeg",
         "-y",  # overwrite output files if they exist
         "-hwaccel", "cuda",             # use CUDA for hardware acceleration
         "-hwaccel_output_format", "cuda",
         "-i", input_path,              # input file
         "-vf", crop_filter,            # apply cropping and scaling
         "-c:v", "h264_nvenc",          # use NVIDIA's encoder
         output_path
    ]
    subprocess.run(ffmpeg_cmd, check=True)

# ---------------------------
# Step 2: Process Preprocessed Videos into a Tensor
# ---------------------------

# Define output directory for tensors
tensor_output_dir = os.path.join("processed_output", "tensors")
os.makedirs(tensor_output_dir, exist_ok=True)

# Define the transformation pipeline (normalization, etc.)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
])

def process_frame(frame):
    # The frame from the preprocessed video is already cropped and scaled.
    # Convert from BGR (OpenCV default) to RGB.
    if len(frame.shape) == 2:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    else:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    tensor_img = transform(pil_img)
    return tensor_img

def sample_epoch_frames(epoch_frames, num_samples=16):
    total_frames = len(epoch_frames)
    indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
    return [epoch_frames[i] for i in indices]

all_epoch_clips = []

# Process each preprocessed video file (assumed to be 40 minutes long)
print("Processing preprocessed videos to extract epochs...")
for video_file in tqdm(os.listdir(preprocessed_folder), desc="Processing videos"):
    if not video_file.endswith('.mp4'):
        continue
    video_path = os.path.join(preprocessed_folder, video_file)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    epoch_duration = 30  # seconds per epoch
    frames_per_epoch = int(fps * epoch_duration)
    
    clip_frames = []
    frame_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        tensor_frame = process_frame(frame)
        clip_frames.append(tensor_frame)
        frame_counter += 1
        
        # Once an epoch (30 seconds) worth of frames is collected, sample 16 frames.
        if frame_counter % frames_per_epoch == 0 and len(clip_frames) > 0:
            epoch_clip = sample_epoch_frames(clip_frames, num_samples=16)
            clip_tensor = torch.stack(epoch_clip)  # shape [16, 3, 224, 224]
            all_epoch_clips.append(clip_tensor)
            clip_frames = []
    
    cap.release()

# Combine all epoch clips into one tensor representing the entire night.
if len(all_epoch_clips) > 0:
    # Final tensor shape: [num_epochs, 16, 3, 224, 224]
    night_tensor = torch.stack(all_epoch_clips)
    tensor_save_path = os.path.join(tensor_output_dir, "night_tensor.pt")
    torch.save(night_tensor, tensor_save_path)
    print(f"Saved night tensor with shape {night_tensor.shape} to '{tensor_save_path}'")
else:
    print("No epochs were processed.")
