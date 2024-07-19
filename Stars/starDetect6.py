import cv2
import numpy as np

# Function to create a circular mask
def create_circular_mask(h, w, center=None, radius=None):
    if center is None:  # use the middle of the image
        center = (int(w / 2), int(h / 2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask

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
        
        # Avoid division by zero in circularity calculation
        if perimeter != 0:
            circularity = 4 * np.pi * (area / (perimeter ** 2))
        else:
            circularity = 0
        

        # Print contour properties for debugging
        #print(f"Contour {i}: Vertices = {len(approx)}, Area = {area}, Aspect Ratio = {aspect_ratio:.2f}, Circularity = {circularity:.2f}")
        
        # Filter out circles based on aspect ratio and circularity, but ensure contour 0 is included
        if (len(approx) >= 8 and len(approx) <= 12 and (10 < area) and 0.8 < aspect_ratio < 1.2 and circularity < 0.3):

            # Check if the area around the contour center is black
            center = (x + w // 2, y + h // 2)
            radius = min(w, h) // 4  # Radius as a fraction of the bounding box size
            mask = create_circular_mask(gray.shape[0], gray.shape[1], center, radius)
            mean_val = cv2.mean(gray, mask.astype(np.uint8) * 255)[0]

            if mean_val < 80:
                potential_stars.append((center, approx))
                # Draw the detected star
                cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
                cv2.putText(image, f"({center[0]},{center[1]})", (center[0], center[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                cv2.putText(image, f"Contour {i}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                print(f"Contour {i}: Vertices = {len(approx)}, Area = {area}, Aspect Ratio = {aspect_ratio:.2f}, Circularity = {circularity:.2f}, Mean Val = {mean_val}")
                print(f"Detected star at: ({center[0]}, {center[1]})")

    return potential_stars, image

# Function to identify star positions and remove duplicates
def identify_star_positions(stars):
    if len(stars) < 4:
        raise Exception("Not enough stars detected.")

    # Remove duplicates based on proximity
    unique_stars = []
    for star in stars:
        if not any(np.linalg.norm(np.array(star[0]) - np.array(uniq_star[0])) < 100 for uniq_star in unique_stars):
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

# Function to detect the color of a piece in a square
def detect_piece_color(square):
    if square.size == 0:
        print("Empty cell detected.")
        return 0  # No piece or white

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(square, cv2.COLOR_BGR2HSV)

    # Define color ranges for red and blue pieces
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    lower_blue = np.array([90, 50, 0])
    upper_blue = np.array([140, 255, 255])

    # Create masks for each color
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # Debugging: Show the masks
    cv2.imshow("Red Mask", mask_red)
    cv2.imshow("Blue Mask", mask_blue)
    cv2.waitKey(100)  # Adjust the delay as needed

    # Detect the color based on the masks
    #print("Red val: " + str(cv2.countNonZero(mask_red)))
    if cv2.countNonZero(mask_red) > 3000:  # Threshold can be adjusted
        return 1  # Red piece
    print("Blue val: " + str(cv2.countNonZero(mask_blue)))
    if cv2.countNonZero(mask_blue) > 3000:
        return 2  # Blue piece
    return 0  # No piece or white

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
image_path = '/projects/connectFour/TestImages/Star/testBoard54.jpg'
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
width = cell_width * 9
height = cell_height * 8

# Start half a cell width and half a cell height away from the corner stars
pts2 = np.float32([
    [cell_width / 2, cell_height / 2], 
    [width - cell_width / 2, cell_height / 2], 
    [width - cell_width / 2, height - cell_height / 2], 
    [cell_width / 2, height - cell_height / 2]
])

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
for i in range(1, 9):
    cv2.line(warped, (i * cell_width, 0), (i * cell_width, height), (255, 0, 0), 2)
for j in range(1, 8):
    cv2.line(warped, (0, j * cell_height), (width, j * cell_height), (255, 0, 0), 2)

# Define the size of each cell in the grid
rows, cols = 6, 7

# Initialize the board state matrix
board_state = np.zeros((rows, cols), dtype=int)

# Loop through each cell in the grid
for i in range(rows):
    for j in range(cols):
        x = (1+j) * cell_width
        y = (1+i) * cell_height
        print(f"Processing cell at position ({i}, {j}) with coordinates ({x}, {y}) and dimensions ({cell_width}, {cell_height})")
        cell = warped[y:y + cell_height, x:x + cell_width]
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
