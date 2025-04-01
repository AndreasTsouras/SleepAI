import cv2
import os

# Define the crop parameters (adjust these for your setup)
bed_x, bed_y, bed_w, bed_h = 100, 200, 300, 200

# Path to your video file (update this to your actual video file path)
video_path = "path/to/your/video.asf"

# Open the video file
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video file!")
    exit()

# Read the first frame from the video
ret, frame = cap.read()
if not ret:
    print("Error: Could not read frame from video!")
    cap.release()
    exit()

# Crop the frame: frame[y:y+h, x:x+w]
cropped_frame = frame[bed_y:bed_y+bed_h, bed_x:bed_x+bed_w]

# Save the cropped frame as an image
output_image = "cropped_frame.jpg"
cv2.imwrite(output_image, cropped_frame)

# Get the absolute path of the saved image and current working directory
abs_path = os.path.abspath(output_image)
cwd = os.getcwd()

print(f"Cropped frame saved to {abs_path}")
print(f"Current working directory: {cwd}")

cap.release()
