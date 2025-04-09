import os
import re
import subprocess
import concurrent.futures
import math

# ---------------------
# PARAMETERS
# ---------------------
BASE_VIDEO_DIR = r"/mnt/d/Student_Video_project"  # Root folder with patient subfolders.
CROP_FILE = "crop_values.txt"  # Crop file with lines like: 1:bl:250,350, br:550,550, tl:620,75, tr:800,100
EPOCH_DURATION = 30            # seconds per epoch
SAMPLE_COUNT = 8              # Desired number of output frames per epoch
FPS = 25                       # Video frame rate (assumed)
extraction_fps = SAMPLE_COUNT / EPOCH_DURATION  # ≈ 0.5333 fps

TARGET_RESOLUTION = "224:224"  # ffmpeg scale filter target (width:height)
# Destination coordinates for perspective transform: map to 224x224
dest_coords = "0:0:224:0:224:224:0:224"

# Path to your GPU-enabled ffmpeg binary (must be the full path)
ffmpeg_path = "/usr/bin/ffmpeg"  # <-- Replace with your GPU-enabled ffmpeg binary path

# Output folder for extracted frames (you can later process these or use them directly)
output_base_folder = "ffmpeg_extracted_frames"
os.makedirs(output_base_folder, exist_ok=True)

# ---------------------
# STEP 1: Read Crop Values from File
# ---------------------
# Crop file format: "1:bl:250,350, br:550,550, tl:620,75, tr:800,100"
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
        else:
            print(f"Line did not match expected format: {line}")
print("Loaded crop data for patients:", list(crop_dict.keys()))

# ---------------------
# Helper: Build an ffmpeg command for a given video file and crop parameters.
# ---------------------
def build_ffmpeg_command(video_path, out_folder, perspective_filter):
    """
    Build an ffmpeg command for the video file.
      - video_path: full path to input video
      - out_folder: folder to save output frames (will be created if needed)
      - perspective_filter: string filter for the perspective transform (with hwdownload, format, etc.)
    
    The command uses:
        -hwaccel cuda -hwaccel_output_format cuda
        -vf: first, hwdownload, format=nv12, then perspective filter, then fps filter.
    """
    os.makedirs(out_folder, exist_ok=True)
    # Output pattern: use the base name of the video file.
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    output_pattern = os.path.join(out_folder, f"{video_basename}_frame_%05d.png")
    
    # Build the filter chain:
    # We first use hwdownload and format=nv12 to bring the frame from GPU,
    # then apply the perspective transform using our filter parameters,
    # then apply the fps filter to sample frames,
    # and optionally scale is already done in perspective if destination is set.
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
# Helper: For a given patient folder, build perspective filter string from crop_dict.
# ---------------------
def build_perspective_filter(patient_number):
    """
    For a given patient number (as 3-digit string), build the ffmpeg perspective filter
    parameter string that maps the input (source) corners to a 224x224 rectangle.
    
    For ffmpeg's perspective filter, the syntax is:
      perspective=in_x0:in_y0:in_x1:in_y1:in_x2:in_y2:in_x3:in_y3:out_x0:out_y0:out_x1:out_y1:out_x2:out_y2:out_x3:out_y3
    We use:
      in: top-left, top-right, bottom-right, bottom-left  (reorder accordingly from crop file)
      out: 0:0, 224:0, 224:224, 0:224
    """
    if patient_number not in crop_dict:
        return None
    coords = crop_dict[patient_number]
    # Reorder: top-left, top-right, bottom-right, bottom-left.
    tl = coords['tl']
    tr = coords['tr']
    br = coords['br']
    bl = coords['bl']
    # Build the filter parameters (as integers separated by colons)
    perspective_params = f"{tl[0]}:{tl[1]}:{tr[0]}:{tr[1]}:{bl[0]}:{bl[1]}:{br[0]}:{br[1]}"
    return perspective_params

# ---------------------
# Helper: Process videos in batches concurrently.
# ---------------------
def process_videos_for_patient(patient_folder, patient_number):
    """
    For a given patient folder, process all .asf video files concurrently (in batches of 3).
    Uses ffmpeg commands that include the perspective transform and fps filtering.
    """
    video_files = [f for f in os.listdir(patient_folder) if f.lower().endswith('.asf')]
    video_files.sort()
    if not video_files:
        print(f"No video files found in {patient_folder}.")
        return

    perspective_filter = build_perspective_filter(patient_number)
    if not perspective_filter:
        print(f"No perspective filter for patient {patient_number}.")
        return
    
    # Create an output subfolder for this patient’s extracted frames.
    patient_frame_folder = os.path.join(output_base_folder, f"patient_{patient_number}_frames")
    os.makedirs(patient_frame_folder, exist_ok=True)
    
    commands = []
    # Build ffmpeg command for each video file.
    for video_file in video_files:
        video_path = os.path.join(patient_folder, video_file)
        cmd = build_ffmpeg_command(video_path, patient_frame_folder, perspective_filter)
        commands.append(cmd)
    
    # Process commands in batches of 3 concurrently.
    batch_size = 3
    for i in range(0, len(commands), batch_size):
        batch = commands[i:i+batch_size]
        processes = []
        for cmd in batch:
            print("Starting ffmpeg command:")
            print(" ".join(cmd))
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            processes.append(p)
        # Wait for current batch to finish.
        for p in processes:
            stdout, stderr = p.communicate()
            if p.returncode != 0:
                print("ffmpeg error:", stderr)
            else:
                print("ffmpeg finished successfully.")
    
    print(f"Finished processing videos for patient {patient_number}.")
    
# ---------------------
# MAIN: Process each patient folder concurrently (or sequentially) using ffmpeg.
# ---------------------
for folder_name in os.listdir(BASE_VIDEO_DIR):
    patient_folder = os.path.join(BASE_VIDEO_DIR, folder_name)
    if not os.path.isdir(patient_folder):
        continue
    # Expect folder name like "SDRI001_001_video"; extract patient number as second segment.
    parts = folder_name.split('_')
    if len(parts) < 2:
        print(f"Folder '{folder_name}' does not match expected pattern; skipping.")
        continue
    patient_number = parts[1].zfill(3)
    print(f"\nProcessing patient {patient_number} from folder '{folder_name}'")
    if patient_number not in crop_dict:
        print(f"No crop data for patient {patient_number}; skipping folder.")
        continue
    process_videos_for_patient(patient_folder, patient_number)

print("All processing complete.")
