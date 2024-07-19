import matplotlib.pyplot as plt
import numpy as np
import cv2

# Function to generate a 5-pointed star
def generate_star_image(filename):
    # Create a blank image
    size = 200
    image = np.zeros((size, size, 3), np.uint8)
    image.fill(255)  # White background

    # Coordinates for the 5-pointed star
    star_points = np.array([
        [100, 10], [120, 70], [180, 70], 
        [130, 110], [150, 170], [100, 130], 
        [50, 170], [70, 110], [20, 70], 
        [80, 70]
    ], np.int32)

    star_points = star_points.reshape((-1, 1, 2))
    
    # Draw the star
    cv2.polylines(image, [star_points], isClosed=True, color=(0, 0, 0), thickness=2)
    cv2.fillPoly(image, [star_points], color=(0, 0, 0))

    # Save the image
    cv2.imwrite(filename, image)

# Generate star images for each corner
generate_star_image('top_left_star.png')
generate_star_image('top_right_star.png')
generate_star_image('bottom_left_star.png')
generate_star_image('bottom_right_star.png')
