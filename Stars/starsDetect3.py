import cv2
import numpy as np

# Function to detect potential stars in the image
def detect_potential_stars(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny edge detection
    edges = cv2.Canny(blurred, 20, 80)
    
    # Resize and visualize edges for debugging
    resized_edges = resize_image(edges, width=800)
    cv2.imshow("Edges", resized_edges)
    cv2.waitKey(0)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    potential_stars = []

    for i, contour in enumerate(contours):
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * (area / (perimeter ** 2))
        
        # Print contour properties for debugging
        print(f"Contour {i}: Vertices = {len(approx)}, Area = {area}, Aspect Ratio = {aspect_ratio:.2f}, Circularity = {circularity:.2f}")
        
        # Filter out circles based on aspect ratio and circularity, but ensure contour 0 is included
        if (len(approx) >= 8 and len(approx) <= 12 and area > 20 and 0.8 < aspect_ratio < 1.2 and circularity < 0.8) or i == 0:
            center = (x + w // 2, y + h // 2)
            potential_stars.append((center, approx))
            # Draw the detected star
            cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
            cv2.putText(image, f"({center[0]},{center[1]})", (center[0], center[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.putText(image, f"Contour {i}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            print(f"Detected star at: ({center[0]}, {center[1]})")

    return potential_stars, image

# Function to identify star positions and remove duplicates
def identify_star_positions(stars):
    if len(stars) < 4:
        raise Exception("Not enough stars detected.")

    # Remove duplicates based on proximity
    unique_stars = []
    for star in stars:
        if not any(np.linalg.norm(np.array(star[0]) - np.array(uniq_star[0])) < 10 for uniq_star in unique_stars):
            unique_stars.append(star)

    if len(unique_stars) < 4:
        raise Exception("Not enough unique stars detected.")
    
    # Sort stars by y-coordinate first, then by x-coordinate
    unique_stars = sorted(unique_stars, key=lambda s: (s[0][1], s[0][0]))
    
    # Identify the top two and bottom two stars
    top_stars = unique_stars[:2]
    bottom_stars = unique_stars[2:]
    
    # Sort top and bottom stars by x-coordinate
    top_stars = sorted(top_stars, key=lambda s: s[0][0])
    bottom_stars = sorted(bottom_stars, key=lambda s: s[0][0])
    
    # Assign positions
    top_left = top_stars[0]
    top_right = top_stars[1]
    bottom_left = bottom_stars[0]
    bottom_right = bottom_stars[1]

    return {
        'top_left': top_left[0],
        'top_right': top_right[0],
        'bottom_left': bottom_left[0],
        'bottom_right': bottom_right[0]
    }

# Function to resize image for better visualization
def resize_image(image, width=None, height=None):
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is not None:
        ratio = width / float(w)
        dim = (width, int(h * ratio))
    else:
        ratio = height / float(h)
        dim = (int(w * ratio), height)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized

# Load an image from file
image_path = '/projects/connectFour/testBoard5.jpg'
image = cv2.imread(image_path)

# Check if image is loaded correctly
if image is None:
    raise Exception(f"Could not open or find the image: {image_path}")

# Detect potential stars
potential_stars, output_image = detect_potential_stars(image)

# Identify star positions relative to each other
star_positions = identify_star_positions(potential_stars)

# Ensure all four corners are detected
if len(star_positions) != 4:
    print("Detected star positions:", star_positions)
    resized_output = resize_image(output_image, width=800)  # Resize for better viewing
    cv2.imshow("Stars Detected", resized_output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    raise Exception("Exactly four stars are required at the corners of the board.")

# Get the corner positions
top_left = star_positions['top_left']
top_right = star_positions['top_right']
bottom_left = star_positions['bottom_left']
bottom_right = star_positions['bottom_right']

# Print star positions for debugging
print("Star positions:")
print(f"Top-left: {top_left}")
print(f"Top-right: {top_right}")
print(f"Bottom-left: {bottom_left}")
print(f"Bottom-right: {bottom_right}")

# Display the original image with detected stars and contours
resized_output = resize_image(output_image, width=800)  # Resize for better viewing
cv2.imshow("Stars and Contours Detected", resized_output)
cv2.waitKey(0)
cv2.destroyAllWindows()
