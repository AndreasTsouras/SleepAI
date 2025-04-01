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

# Initialize the HOG person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Iterate over frames to find one where no patient is detected
frame_found = False
frame_index = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    frame_index += 1

    # Crop the frame to the bed region
    cropped_frame = frame[bed_y:bed_y+bed_h, bed_x:bed_x+bed_w]
    
    # Optionally convert to grayscale for detection
    # (HOG detector can work on color images, but sometimes grayscale works better)
    gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
    
    # Detect people in the cropped region.
    # You may need to adjust parameters like winStride, padding, and scale
    rects, weights = hog.detectMultiScale(cropped_frame, winStride=(8,8), padding=(8,8), scale=1.05)
    
    # If no detections, assume no patient is in bed.
    if len(rects) == 0:
        output_image = f"empty_bed_frame_{frame_index}.jpg"
        cv2.imwrite(output_image, cropped_frame)
        print(f"Saved an empty bed frame to {os.path.abspath(output_image)}")
        frame_found = True
        break

cap.release()

if not frame_found:
    print("No frame without a detected patient was found in the video.")
