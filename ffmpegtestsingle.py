import os
import subprocess
import concurrent.futures

# ---------------------
# Configuration
# ---------------------
# Replace with the full path to your GPU-enabled ffmpeg executable.
ffmpeg_path = "/mnt/d/Andrew-SleepAI/ffmpeg_gpu/ffmpeg"

# List of video files (for example, parts of one clip).
# Ensure these files are in your working directory or provide full paths.
video_files = [
    "video_part1.asf",
    "video_part2.asf",
    "video_part3.asf"
]

# Folder where ffmpeg will save extracted frames.
output_folder = "test_extracted_frames"
os.makedirs(output_folder, exist_ok=True)

# Desired output frame resolution (width:height).
target_resolution = "224:224"

# For a 30-second epoch at 25 fps, there are 750 frames.
# To extract 16 frames uniformly, set the extraction fps to 16/30 ≈ 0.5333 fps.
extraction_fps = 16 / 30  # approximately 0.5333

# Crop/Perspective parameters:
# We assume your crop file gave you the following correct order of corners:
# Top-left: (620, 75)
# Top-right: (800, 100)
# Bottom-right: (550, 550)
# Bottom-left: (250, 350)
# And you want to map these to a 224x224 rectangle:
#   Out TL: (0,0), Out TR: (224,0), Out BR: (224,224), Out BL: (0,224)
# The ffmpeg perspective filter expects:
#   perspective=in_x0:in_y0:in_x1:in_y1:in_x2:in_y2:in_x3:in_y3:out_x0:out_y0:out_x1:out_y1:out_x2:out_y2:out_x3:out_y3:eval=init
# So the filter string should be:
perspective_filter = "620:75:800:100:550:550:250:350:0:0:224:0:224:224:0:224:eval=init"

# ---------------------
# Helper: Build an ffmpeg command for one video file.
# ---------------------
def build_ffmpeg_command(video_file, out_folder, perspective_filter, extraction_fps, target_resolution):
    # Build an output pattern that includes the video file base name.
    base_name = os.path.splitext(os.path.basename(video_file))[0]
    output_pattern = os.path.join(out_folder, f"{base_name}_frame_%05d.png")
    
    # Construct filter chain:
    # We use hwdownload, format=nv12 to convert GPU frames to CPU,
    # then apply the perspective crop/transform,
    # then fps to sample frames,
    # then scale to target resolution.
    filter_chain = f"hwdownload,format=nv12,perspective={perspective_filter},fps={extraction_fps},scale={target_resolution}"
    
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
# Process the video files concurrently in batches of 3.
# ---------------------
def process_batch(commands):
    processes = []
    for cmd in commands:
        print("Starting ffmpeg command:")
        print(" ".join(cmd))
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        processes.append(p)
    # Wait for all processes in this batch to complete.
    for p in processes:
        stdout, stderr = p.communicate()
        if p.returncode != 0:
            print("Error in ffmpeg command:")
            print(stderr)
        else:
            print("Command completed successfully.")
            
# Run all commands in batches (here, batch size is 3)
batch_size = 3
all_commands = [build_ffmpeg_command(vf, output_folder, perspective_filter, extraction_fps, target_resolution) for vf in video_files]

# Process all commands concurrently in batches.
for i in range(0, len(all_commands), batch_size):
    batch = all_commands[i:i+batch_size]
    process_batch(batch)

print("All ffmpeg commands finished.")
