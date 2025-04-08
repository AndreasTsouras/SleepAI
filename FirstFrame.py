import os
import cv2

# Base directory with patient folders.
base_dir = r"/mnt/d/Student_Video_project"

# Folder where we will save the first frames.
output_folder = "first_frames"
os.makedirs(output_folder, exist_ok=True)

# Iterate over each item in the base directory.
for folder_name in os.listdir(base_dir):
    patient_folder_path = os.path.join(base_dir, folder_name)
    if not os.path.isdir(patient_folder_path):
        continue  # Skip if not a directory.

    # Extract the patient number from the folder name.
    # For example, if folder is "SDRI001_001_video", then splitting by '_' gives:
    # ['SDRI001', '001', 'video'] and we take the second element as the patient number.
    parts = folder_name.split('_')
    if len(parts) < 2:
        print(f"Folder '{folder_name}' does not match expected pattern; skipping.")
        continue
    patient_number = parts[1]

    # Look for video files in this folder (assuming .asf files; case-insensitive).
    video_files = [f for f in os.listdir(patient_folder_path) if f.lower().endswith('.asf')]
    if not video_files:
        print(f"No video files found in folder '{folder_name}'.")
        continue

    # Optionally sort the video files and take the first one.
    video_files.sort()
    video_path = os.path.join(patient_folder_path, video_files[0])
    print(f"Processing video for patient {patient_number} at: {video_path}")

    # Open the video file.
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        continue

    # Read the first frame.
    ret, frame = cap.read()
    if not ret:
        print(f"Error: Could not read the first frame from {video_path}")
        cap.release()
        continue

    # Define the output image filename.
    output_image_path = os.path.join(output_folder, f"patient_{patient_number}_first_frame.jpg")
    cv2.imwrite(output_image_path, frame)
    print(f"Saved first frame for patient {patient_number} to {output_image_path}")

    cap.release()

print("Processing complete.")
