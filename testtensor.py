import cv2
import numpy as np

# Load the saved first frame image (update the filename as needed)
image = cv2.imread("first_frame.jpg")
if image is None:
    print("Error: Image not found. Please check the filename and path.")
    exit()

# Define the source points from the four corners provided.
# We assume the points need to be ordered as [top-left, top-right, bottom-right, bottom-left].
src_pts = np.float32([
    [620, 75],    # top-left
    [800, 100],   # top-right
    [550, 550],   # bottom-right
    [250, 350]    # bottom-left
])

# Define the destination points in a rectangular shape.
# Here we've chosen width = 361 and height = 515 based on our approximate calculations.
dst_pts = np.float32([
    [0, 0],        # top-left
    [361, 0],      # top-right
    [361, 515],    # bottom-right
    [0, 515]       # bottom-left
])

# Compute the perspective transform matrix.
M = cv2.getPerspectiveTransform(src_pts, dst_pts)

# Apply the perspective warp using the computed matrix.
warped = cv2.warpPerspective(image, M, (361, 515))

# Save the warped image to a file so you can inspect it.
output_filename = "warped_bed.jpg"
cv2.imwrite(output_filename, warped)
print(f"Warped image saved as {output_filename}")
