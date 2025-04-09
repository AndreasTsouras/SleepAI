def build_ffmpeg_command(video_file, out_folder, perspective_filter, extraction_fps, target_resolution, gpu_id):
    base_name = os.path.splitext(os.path.basename(video_file))[0]
    output_pattern = os.path.join(out_folder, f"{base_name}_frame_%05d.png")
    
    # Create filter chain.
    filter_chain = f"hwdownload,format=nv12,perspective={perspective_filter},fps={extraction_fps},scale={target_resolution}"
    
    command = [
        ffmpeg_path,
        "-hwaccel", "cuda",
        "-hwaccel_device", str(gpu_id),  # Select specific GPU device.
        "-hwaccel_output_format", "cuda",
        "-i", video_file,
        "-vf", filter_chain,
        "-vsync", "0",
        output_pattern
    ]
    return command

# Example of assigning each video file a specific GPU:
gpu_ids = [0, 1, 2]  # indexes for your three RTX 3080 GPUs

all_commands = []
for i, video in enumerate(video_files):
    gpu_id = gpu_ids[i % len(gpu_ids)]
    all_commands.append(build_ffmpeg_command(video, output_folder, perspective_filter, extraction_fps, target_resolution, gpu_id))
