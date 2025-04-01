import cv2
import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm  # Progress bar library

# Define output directories for tensors and video.
output_dir = "processed_output"
tensor_dir = os.path.join(output_dir, "tensors")
video_dir = os.path.join(output_dir, "videos")
os.makedirs(tensor_dir, exist_ok=True)
os.makedirs(video_dir, exist_ok=True)

# Example coordinates for the bedframe region (adjust these for your setup)
bed_x, bed_y, bed_w, bed_h = 100, 200, 300, 200

# Define a transformation pipeline.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
])

def process_frame(frame):
    # Crop to the bedframe region
    cropped = frame[bed_y:bed_y+bed_h, bed_x:bed_x+bed_w]
    # Resize to 224x224
    resized = cv2.resize(cropped, (224, 224))
    
    # Convert for tensor processing:
    if len(resized.shape) == 2:
        resized_rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    else:
        resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # For display, convert the RGB image back to BGR
    display_frame = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2BGR)
    
    pil_img = Image.fromarray(resized_rgb)
    tensor_img = transform(pil_img)
    return tensor_img, display_frame

def sample_epoch_frames(epoch_frames, num_samples=16):
    total_frames = len(epoch_frames)
    indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
    return [epoch_frames[i] for i in indices]

# Folder containing the .asf files (update to your WSL path)
folder_path = "/mnt/e/SDRI001_001_video"

video_files = [f for f in os.listdir(folder_path) if f.endswith('.asf')]
video_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

all_epoch_clips = []
video_writer = None

# Use tqdm to iterate over video files
for video_file in tqdm(video_files, desc="Processing video files"):
    video_path = os.path.join(folder_path, video_file)
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    epoch_duration = 30  # seconds
    frames_per_epoch = int(fps * epoch_duration)
    
    clip_frames = []
    frame_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_tensor, display_frame = process_frame(frame)
        clip_frames.append(processed_tensor)
        frame_counter += 1

        if video_writer is None:
            output_video_path = os.path.join(video_dir, "processed_night.avi")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (224, 224))
        
        video_writer.write(display_frame)

        if frame_counter % frames_per_epoch == 0:
            epoch_clip = sample_epoch_frames(clip_frames, num_samples=16)
            clip_tensor = torch.stack(epoch_clip)
            all_epoch_clips.append(clip_tensor)
            clip_frames = []
    
    cap.release()

if video_writer is not None:
    video_writer.release()

if len(all_epoch_clips) > 0:
    night_tensor = torch.stack(all_epoch_clips)
    tensor_save_path = os.path.join(tensor_dir, "night_tensor.pt")
    torch.save(night_tensor, tensor_save_path)
    print(f"Saved night tensor with shape {night_tensor.shape} to '{tensor_save_path}'")
else:
    print("No epochs were processed.")
