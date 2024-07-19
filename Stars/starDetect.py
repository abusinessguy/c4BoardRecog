import cv2
import numpy as np

# Function to detect potential stars in the image
def detect_potential_stars(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Resize and visualize edges for debugging
    resized_edges = resize_image(edges, width=800)
    cv2.imshow("Edges", resized_edges)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    potential_stars = []

    for contour in contours:
        # Approximate the contour
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Filter based on geometric properties
        if len(approx) == 10 and cv2.contourArea(contour) > 100:  # Adjust conditions as needed
            x, y, w, h = cv2.boundingRect(contour)
            center = (x + w // 2, y + h // 2)
            potential_stars.append((center, approx))
            # Draw the detected star
            cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
            cv2.putText(image, f"({center[0]},{center[1]})", (center[0], center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return potential_stars, image

# Function to identify star positions relative to each other
def identify_star_positions(stars):
    if len(stars) < 4:
        raise Exception("Not enough stars detected.")

    # Sort stars based on their coordinates
    stars = sorted(stars, key=lambda s: (s[0][1], s[0][0]))  # Sort by y first, then x

    # Extract star coordinates
    coords = [star[0] for star in stars]

    # Identify the top-left and bottom-left stars
    top_left = min(coords, key=lambda x: x[0] + x[1])
    bottom_left = max(coords, key=lambda x: x[1] - x[0])

    # Remove identified stars from the list
    coords.remove(top_left)
    coords.remove(bottom_left)

    # Identify the top-right and bottom-right stars
    top_right = min(coords, key=lambda x: x[1] - x[0])
    bottom_right = max(coords, key=lambda x: x[0] + x[1])

    # Return the identified positions
    return {
        'top_left': top_left,
        'top_right': top_right,
        'bottom_left': bottom_left,
        'bottom_right': bottom_right
    }

# Function to detect the color of a piece in a square
def detect_piece_color(square):
    if square.size == 0:
        print("Empty cell detected.")
        return 0  # No piece or white

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(square, cv2.COLOR_BGR2HSV)

    # Define color ranges for red and blue pieces
    lower_red = np.array([0, 70, 50])
    upper_red = np.array([10, 255, 255])
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])

    # Create masks for each color
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # Detect the color based on the masks
    if cv2.countNonZero(mask_red) > 50:  # Threshold can be adjusted
        return 1  # Red piece
    elif cv2.countNonZero(mask_blue) > 50:
        return 2  # Blue piece
    else:
        return 0  # No piece or white

# Function to resize image for better visualization
def resize_image(image, width=None, height=None):
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        ratio = height / float(h)
        dim = (int(w * ratio), height)
    else:
        ratio = width / float(w)
        dim = (width, int(h * ratio))
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

# Calculate the cell width and height based on the positions of the stars
cell_width = int((top_right[0] - top_left[0]) / 8)
cell_height = int((bottom_left[1] - top_left[1]) / 7)

# Print calculated cell width and height for debugging
print(f"Cell width: {cell_width}")
print(f"Cell height: {cell_height}")

# Define points for perspective transform
pts1 = np.float32([top_left, top_right, bottom_right, bottom_left])
width = cell_width * 7
height = cell_height * 6
pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

# Print points used for perspective transform
print("Points for perspective transform:")
print(f"pts1: {pts1}")
print(f"pts2: {pts2}")

# Get the transformation matrix
matrix = cv2.getPerspectiveTransform(pts1, pts2)

# Apply the warp transformation
warped = cv2.warpPerspective(image, matrix, (width, height))

# Check if the warped image is empty
if warped.size == 0:
    print("Warped image is empty.")
    cv2.imshow("Warped Image", warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    raise Exception("Warped image is empty.")

# Draw the grid lines on the warped image for debugging
for i in range(1, 7):
    cv2.line(warped, (i * cell_width, 0), (i * cell_width, height), (255, 0, 0), 2)
for j in range(1, 6):
    cv2.line(warped, (0, j * cell_height), (width, j * cell_height), (255, 0, 0), 2)

# Define the size of each cell in the grid
rows, cols = 6, 7

# Initialize the board state matrix
board_state = np.zeros((rows, cols), dtype=int)

# Loop through each cell in the grid
for i in range(rows):
    for j in range(cols):
        x = j * cell_width
        y = i * cell_height
        print(f"Processing cell at position ({i}, {j}) with coordinates ({x}, {y}) and dimensions ({cell_width}, {cell_height})")
        cell = warped[y:y+cell_height, x:x+cell_width]
        color = detect_piece_color(cell)
        board_state[i, j] = color

# Display the board state
print("Board state:")
print(board_state)

# Display the warped image with grid lines (for visualization)
resized_warped = resize_image(warped, width=800)
cv2.imshow("Warped Image with Grid", resized_warped)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Display the original image with detected stars and contours
resized_output = resize_image(output_image, width=800)  # Resize for better viewing
cv2.imshow("Stars and Contours Detected", resized_output)
cv2.waitKey(0)
cv2.destroyAllWindows()
