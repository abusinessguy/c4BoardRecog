# Tweak the red and blue values based on what TIAGO sees and the percent thresholds
# could implement more logic for 
import cv2
import apriltag
import numpy as np

def preprocess_image(image, reduction):
    # Reduce image resolution to reduce complexity
    width = int(image.shape[1] * reduction)
    height = int(image.shape[0] * reduction)
    dim = (width, height)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    
    return resized_image

def detect_apriltags(image_path, tag_family='tag36h11'):
    # Load the image with AprilTags
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return None, None

    # Preprocess the image
    preprocessed_image = preprocess_image(image, 0.5)

    # Initialize the AprilTag detector options
    options = apriltag.DetectorOptions(families=tag_family)

    # Initialize the AprilTag detector
    detector = apriltag.Detector(options)

    try:
        # Detect AprilTags in the preprocessed image
        detections = detector.detect(preprocessed_image)

        if not detections:
            print("No AprilTags detected.")
            return None, None
        
        # Convert grayscale image to BGR for colored drawing
        image_bgr = cv2.cvtColor(preprocessed_image, cv2.COLOR_GRAY2BGR)

        # Draw detections on the image
        tag_centers = {}
        for detection in detections:
            corners = detection.corners.astype(np.int32).reshape(-1, 1, 2)

            # Draw the tag outline
            cv2.polylines(image_bgr, [corners], isClosed=True, color=(255, 0, 0), thickness=2)
            
            # Calculate the center of the tag
            center = tuple(np.mean(corners, axis=0).astype(int).flatten())
            tag_centers[detection.tag_id] = center

            # Draw the tag ID at the center in blue color
            cv2.putText(image_bgr, str(detection.tag_id), center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Draw vectors from the center to each corner in green color
            for corner in corners:
                cv2.line(image_bgr, center, tuple(corner[0]), (0, 255, 0), 2)

        # Resize image for display
        scale_percent = 50  # Percent of original size
        width = int(image_bgr.shape[1] * scale_percent / 100)
        height = int(image_bgr.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized_image = cv2.resize(image_bgr, dim, interpolation=cv2.INTER_AREA)

        # Show the resized image with detected tags
        cv2.imshow('Detected AprilTags', resized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return tag_centers, image_bgr
    
    except Exception as e:
        print(f"An error occurred during detection: {e}")
        return None, None

# Function to detect the color of a piece in a square
def detect_piece_color(square):
    if square.size == 0:
        return 0

    hsv = cv2.cvtColor(square, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    lower_blue = np.array([90, 50, 0])
    upper_blue = np.array([140, 255, 255])

    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    print("Red val: " + str((cv2.countNonZero(mask_red))/(square.size/3)))
    if (cv2.countNonZero(mask_red))/(square.size/3) > .4:
        return 1  # Red piece
    print("Blue val: " + str(cv2.countNonZero(mask_blue)/(square.size/3)))
    if cv2.countNonZero(mask_blue)/(square.size/3) > .4:
        return 2  # Blue piece
    return 0  # No piece or white

def main():
    # Path to the image
    image_path = '/projects/connectFour/TestImages/AprilTag/wooden6.jpg'

    # Detect AprilTags to find corners
    tag_centers, output_image = detect_apriltags(image_path, tag_family='tag36h11')

    if not tag_centers or len(tag_centers) != 4:
        print("Exactly four AprilTags are required at the corners of the board.")
        if output_image is not None:
            cv2.imshow("AprilTags Detected", output_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    # Identify corner positions
    top_left = tag_centers[0]
    top_right = tag_centers[1]
    bottom_left = tag_centers[2]
    bottom_right = tag_centers[3]

    print("AprilTag positions:")
    print(f"Top-left: {top_left}")
    print(f"Top-right: {top_right}")
    print(f"Bottom-left: {bottom_left}")
    print(f"Bottom-right: {bottom_right}")

    image = cv2.imread(image_path)
    image = preprocess_image(image, 0.5)
 
    cell_width = int((top_right[0] - top_left[0]) / 8)
    cell_height = int((bottom_left[1] - top_left[1]) / 7)

    print(f"Cell width: {cell_width}")
    print(f"Cell height: {cell_height}")

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

    print("Points for perspective transform:")
    print(f"pts1: {pts1}")
    print(f"pts2: {pts2}")

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(image, matrix, (width, height))

    if warped.size == 0:
        print("Warped image is empty.")
        cv2.imshow("Warped Image", warped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # Draw grid lines on the warped image
    for i in range(1, 9):  # Vertical lines
        cv2.line(warped, (i * cell_width, 0), (i * cell_width, height), (255, 0, 0), 2)
    for j in range(1, 8):  # Horizontal lines
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

    cv2.imshow("Warped Image with Grid", warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
