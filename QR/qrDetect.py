import cv2
from pyzbar import pyzbar
import numpy as np

# Function to detect QR codes and visualize them
def detect_qr_codes(image):
    qr_codes = pyzbar.decode(image)
    qr_positions = {}
    
    for qr in qr_codes:
        # Extract the bounding box coordinates of the QR code
        (x, y, w, h) = qr.rect
        # Calculate the center point of the QR code
        center = (x + w // 2, y + h // 2)
        # Get the data from the QR code
        data = qr.data.decode("utf-8")
        qr_positions[data] = center
        
        # Draw a bounding box around the QR code for visualization
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, data, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return qr_positions

# Load an image from file
image_path = '/projects/connectFour/testBoard4.jpg'
image = cv2.imread(image_path)

# Check if image is loaded correctly
if image is None:
    raise Exception(f"Could not open or find the image: {image_path}")

# Detect QR codes
qr_positions = detect_qr_codes(image)

# Ensure all four corners are detected
if len(qr_positions) != 4:
    print("Detected QR codes:", qr_positions)
    raise Exception("Exactly four QR codes are required at the corners of the board.")

# Get the corner positions
top_left = qr_positions['top_left']
top_right = qr_positions['top_right']
bottom_left = qr_positions['bottom_left']
bottom_right = qr_positions['bottom_right']

# Define points for perspective transform
pts1 = np.float32([top_left, top_right, bottom_right, bottom_left])
width, height = 800, 800  # Adjust as needed for your board size
pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

# Get the transformation matrix
matrix = cv2.getPerspectiveTransform(pts1, pts2)

# Apply the warp transformation
warped = cv2.warpPerspective(image, matrix, (width, height))

# Display the warped image (for visualization)
cv2.imshow("Warped Image", warped)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Display the original image with detected QR codes
cv2.imshow("QR Codes Detected", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
