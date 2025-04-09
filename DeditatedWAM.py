import os
import subprocess
import time

# ---------------------
# Configuration
# ---------------------
# Full path to your GPU-enabled ffmpeg executable.
ffmpeg_path = "/mnt/d/Andrew-SleepAI/ffmpeg_gpu/ffmpeg"

# List of video files (for example, parts of one clip).
video_files = [
    "video_part1.asf",
    "video_part2.asf",
    "video_part3.asf"
]

# Folder where ffmpeg will save extracted frames.
output_folder = "test_extracted_frames"
os.makedirs(output_folder, exist_ok=True)

# Desired output resolution (width:height).
target_resolution = "224:224"

# For a 30-second epoch at 25 fps, there are 750 frames.
# Extract 16 frames uniformly, so sample at ≈ 0.5333 fps.
extraction_fps = 16 / 30  # approximately 0.5333

# Crop/Perspective parameters (only using input quadrilateral in this ffmpeg version).
# Four source points: TL (620,75), TR (800,100), BR (550,550), BL (250,350).
perspective_filter = "620:75:800:100:550:550:250:350"

# ---------------------
# Helper: Build ffmpeg command for one video file.
# ---------------------
def build_ffmpeg_command(video_file, out_folder, perspective_filter, extraction_fps, target_resolution):
    base_name = os.path.splitext(os.path.basename(video_file))[0]
    output_pattern = os.path.join(out_folder, f"{base_name}_frame_%05d.png")
    
    # Construct filter chain:
    # - First download the frames from GPU (hwdownload and format conversion),
    # - then apply perspective transform,
    # - sample frames using fps,
    # - and finally scale to the target resolution.
    filter_chain = (
        f"hwdownload,format=nv12,perspective={perspective_filter},"
        f"fps={extraction_fps},scale={target_resolution}"
    )
    
    command = [
        ffmpeg_path,
        "-hwaccel", "cuda",
        "-hwaccel_output_format", "cuda",
        "-i", video_file,
        "-vf", filter_chain,
        "-vsync", "0",
        output_pattern
    ]
    return command

# ---------------------
# Running ffmpeg processes with designated GPUs.
# ---------------------
def run_ffmpeg_processes(commands_with_env):
    processes = []
    for cmd, env in commands_with_env:
        print("Starting command:")
        print(" ".join(cmd))
        # Launch the process.
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        processes.append(p)
        time.sleep(1)  # Optional: slight delay to prevent potential race issues.
    # Wait for all processes to complete.
    for p in processes:
        stdout, stderr = p.communicate()
        if p.returncode != 0:
            print("Error in ffmpeg command:")
            print(stderr)
        else:
            print("ffmpeg command completed successfully.")
            
# ---------------------
# Prepare commands with each process limited to a single GPU.
# ---------------------
gpu_ids = ["0", "1", "2"]  # Your three RTX 3080 GPUs.

commands_with_env = []
for i, video in enumerate(video_files):
    # Assign GPU cyclically.
    gpu_id = gpu_ids[i % len(gpu_ids)]
    cmd = build_ffmpeg_command(video, output_folder, perspective_filter, extraction_fps, target_resolution)
    # Create a copy of the environment and restrict visibility.
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    print(f"Mapping {video} to GPU {gpu_id}")
    commands_with_env.append((cmd, env))

# Run all ffmpeg commands concurrently in one batch.
run_ffmpeg_processes(commands_with_env)

print("All ffmpeg commands finished.")
