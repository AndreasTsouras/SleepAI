import cv2
import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

# Example coordinates for the bedframe region (adjust these for your setup)
bed_x, bed_y, bed_w, bed_h = 100, 200, 300, 200

# Define a transformation pipeline.
# For IR data, you may need to compute your own mean and std. Here we use dummy values.
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts image to tensor and scales pixel values to [0, 1]
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
])

def process_frame(frame):
    # Crop to the bedframe region
    cropped = frame[bed_y:bed_y+bed_h, bed_x:bed_x+bed_w]
    # Resize to 224x224 (required input size for TimeSformer)
    resized = cv2.resize(cropped, (224, 224))
    
    # Check the number of channels in the resized frame:
    if len(resized.shape) == 2:  # grayscale image has shape (H, W)
        print("Frame is grayscale with 1 channel.")
        # Convert grayscale to a 3-channel image by replicating the channel
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    else:
        # Assuming the frame is in BGR format, convert it to RGB.
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # Convert to a PIL Image for transformation
    pil_img = Image.fromarray(resized)
    # Apply transformations (to tensor and normalization)
    tensor_img = transform(pil_img)
    
    # Print out the shape of the tensor (should be [3, 224, 224])
    print("Processed frame tensor shape:", tensor_img.shape)
    return tensor_img

def sample_epoch_frames(epoch_frames, num_samples=16):
    # Uniformly sample num_samples frames from the epoch
    total_frames = len(epoch_frames)
    indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
    return [epoch_frames[i] for i in indices]

# Folder containing the .asf files
folder_path = "path/to/your/folder"  # change this to your folder path

# Get all .asf files and sort them based on the number after the underscore.
video_files = [f for f in os.listdir(folder_path) if f.endswith('.asf')]
video_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

all_epoch_clips = []  # List to store each epoch's tensor

# Iterate through each video file (each representing a 40-minute block)
for video_file in video_files:
    video_path = os.path.join(folder_path, video_file)
    print(f"Processing {video_path}...")
    cap = cv2.VideoCapture(video_path)
    
    # Retrieve FPS (frames per second)
    fps = cap.get(cv2.CAP_PROP_FPS)
    epoch_duration = 30  # seconds per epoch
    frames_per_epoch = int(fps * epoch_duration)
    
    clip_frames = []
    frame_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame (crop, resize, and normalize)
        processed_tensor = process_frame(frame)
        clip_frames.append(processed_tensor)
        frame_counter += 1

        # Once we've accumulated one epoch's worth of frames, sample and form a clip
        if frame_counter % frames_per_epoch == 0:
            epoch_clip = sample_epoch_frames(clip_frames, num_samples=16)
            # Stack frames into a tensor of shape [16, 3, 224, 224]
            clip_tensor = torch.stack(epoch_clip)
            print("Epoch clip tensor shape:", clip_tensor.shape)
            all_epoch_clips.append(clip_tensor)
            # Reset for the next epoch within the same video file
            clip_frames = []
    
    cap.release()

# Combine all epoch clips from all files into one tensor.
if len(all_epoch_clips) > 0:
    # The final tensor shape will be [num_epochs, 16, 3, 224, 224]
    night_tensor = torch.stack(all_epoch_clips)
    torch.save(night_tensor, "night_tensor.pt")
    print(f"Saved night tensor with shape {night_tensor.shape} to 'night_tensor.pt'")
else:
    print("No epochs were processed.")
