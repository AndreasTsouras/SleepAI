import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from decord import VideoReader, gpu
from tqdm import tqdm

# Example coordinates for the bedframe region (adjust these for your setup)
bed_x, bed_y, bed_w, bed_h = 100, 200, 300, 200

# Define a transformation pipeline.
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts PIL image to tensor and scales pixel values to [0, 1]
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
])

def process_frame(frame):
    # frame is a decord NDArray, convert to numpy array
    frame_np = frame.asnumpy()  # shape: (H, W, 3)
    # Crop the frame to the bed region
    cropped = frame_np[bed_y:bed_y+bed_h, bed_x:bed_x+bed_w, :]
    # Convert cropped frame to a PIL image and resize to 224x224
    pil_img = Image.fromarray(cropped.astype('uint8'))
    pil_img = pil_img.resize((224, 224))
    # Apply the transformation pipeline
    tensor_img = transform(pil_img)
    return tensor_img

def sample_epoch_frames(epoch_frames, num_samples=16):
    total_frames = len(epoch_frames)
    indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
    return [epoch_frames[i] for i in indices]

# Folder containing the .asf files (update this to your WSL path)
folder_path = "/mnt/e/SDRI001_001_video"

# Get all .asf files and sort them by the number after the underscore.
video_files = [f for f in os.listdir(folder_path) if f.endswith('.asf')]
video_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

all_epoch_clips = []  # List to store each epoch's tensor

print("Processing videos with decord...")
for video_file in tqdm(video_files, desc="Processing videos"):
    video_path = os.path.join(folder_path, video_file)
    # Create a VideoReader using GPU if available
    try:
        vr = VideoReader(video_path, ctx=gpu(0))
    except Exception as e:
        print(f"Could not use GPU for {video_file}, falling back to CPU. Error: {e}")
        from decord import cpu
        vr = VideoReader(video_path, ctx=cpu(0))
    
    # Get FPS and calculate how many frames correspond to 30 seconds
    fps = vr.get_avg_fps()
    frames_per_epoch = int(fps * 30)  # 30-second epoch
    
    clip_frames = []
    frame_counter = 0

    # Iterate over all frames in the video
    for frame in vr:
        tensor_frame = process_frame(frame)
        clip_frames.append(tensor_frame)
        frame_counter += 1

        # Once an epoch's worth of frames is collected, sample and form a clip.
        if frame_counter % frames_per_epoch == 0 and len(clip_frames) > 0:
            epoch_clip = sample_epoch_frames(clip_frames, num_samples=16)
            clip_tensor = torch.stack(epoch_clip)  # shape: [16, 3, 224, 224]
            all_epoch_clips.append(clip_tensor)
            clip_frames = []

    # Process any remaining frames in the video that constitute an epoch.
    if len(clip_frames) >= 16:
        epoch_clip = sample_epoch_frames(clip_frames, num_samples=16)
        clip_tensor = torch.stack(epoch_clip)
        all_epoch_clips.append(clip_tensor)

if len(all_epoch_clips) > 0:
    # Combine all epoch clips into one tensor with shape: [num_epochs, 16, 3, 224, 224]
    night_tensor = torch.stack(all_epoch_clips)
    torch.save(night_tensor, "night_tensor.pt")
    print(f"Saved night tensor with shape {night_tensor.shape} to 'night_tensor.pt'")
else:
    print("No epochs were processed.")
