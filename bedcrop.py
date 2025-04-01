import cv2

# Example coordinates for the bedframe region in the original frame.
# Adjust these values (x, y, width, height) to match your video.
bed_x, bed_y, bed_w, bed_h = 100, 200, 300, 200

# Open the input video file.
cap = cv2.VideoCapture("input_video.mp4")

# Get the frames per second (fps) from the input video.
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the output video writer with resolution 224x224.
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("cropped_output.mp4", fourcc, fps, (224, 224))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Crop the frame to the bedframe region.
    bed_region = frame[bed_y:bed_y+bed_h, bed_x:bed_x+bed_w]

    # Resize the cropped region to 224x224 pixels (input size for TimeSformer).
    bed_region_resized = cv2.resize(bed_region, (224, 224))

    # Write the processed frame to the output video.
    out.write(bed_region_resized)

cap.release()
out.release()
