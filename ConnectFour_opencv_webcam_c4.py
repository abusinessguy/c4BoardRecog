import time
import random
import numpy as np
import apriltag
import cv2

def preprocess_image(image, reduction):
    """Reduce image resolution to reduce complexity."""
    width = int(image.shape[1] * reduction)
    height = int(image.shape[0] * reduction)
    dim = (width, height)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized_image

def detect_apriltags(image, tag_family='tag36h11', reduction=0.5):
    """Detect AprilTags in the given image."""
    preprocessed_image = preprocess_image(image, reduction)
    options = apriltag.DetectorOptions(families=tag_family)
    detector = apriltag.Detector(options)

    try:
        detections = detector.detect(preprocessed_image)
        if not detections:
            return None, None

        image_bgr = cv2.cvtColor(preprocessed_image, cv2.COLOR_GRAY2BGR)
        tag_centers = {}
        for detection in detections:
            corners = detection.corners.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(image_bgr, [corners], isClosed=True, color=(255, 0, 0), thickness=2)
            center = tuple(np.mean(corners, axis=0).astype(int).flatten())
            tag_centers[detection.tag_id] = center
            cv2.putText(image_bgr, str(detection.tag_id), center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            for corner in corners:
                cv2.line(image_bgr, center, tuple(corner[0]), (0, 255, 0), 2)

        return tag_centers, image_bgr

    except Exception as e:
        print(f"An error occurred during detection: {e}")
        return None, None

def detect_piece_color(square):
    """Detect the color of a piece in the given square."""
    if square.size == 0:
        return 0

    try:
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

        if (cv2.countNonZero(mask_red)) / (square.size / 3) > 0.4:
            return 1  # Red piece
        if cv2.countNonZero(mask_blue) / (square.size / 3) > 0.4:
            return 2  # Blue piece
        return 0  # No piece or white
    except Exception as e:
        print(f"An error occurred during piece color detection: {e}")
        return 0

def display_image_with_grid(warped, cell_width, cell_height, rows=8, cols=9):
    """Draw grid lines on the warped image and display it."""
    for i in range(1, cols):  # Vertical lines
        cv2.line(warped, (i * cell_width, 0), (i * cell_width, cell_height * rows), (255, 0, 0), 2)
    for j in range(1, rows):  # Horizontal lines
        cv2.line(warped, (0, j * cell_height), (cell_width * cols, j * cell_height), (255, 0, 0), 2)

    cv2.imshow("Warped Image with Grid", warped)
    cv2.waitKey(1)  # Use a short delay for real-time display

def calculate_perspective_transform(top_left, top_right, bottom_left, bottom_right, cell_width, cell_height):
    """Calculate the perspective transform matrix."""
    pts1 = np.float32([top_left, top_right, bottom_right, bottom_left])
    width = cell_width * 9
    height = cell_height * 8
    pts2 = np.float32([
        [cell_width / 2, cell_height / 2],
        [width - cell_width / 2, cell_height / 2],
        [width - cell_width / 2, height - cell_height / 2],
        [cell_width / 2, height - cell_height / 2]
    ])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    return matrix, width, height

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    try:
        while True:
            status, frame = cap.read()
            if not status:
                print("Error: Failed to capture image")
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            tag_centers, output_image = detect_apriltags(gray_frame)

            if tag_centers and len(tag_centers) == 4:
                top_left = tag_centers[0]
                top_right = tag_centers[1]
                bottom_left = tag_centers[2]
                bottom_right = tag_centers[3]

                cell_width = int((top_right[0] - top_left[0]) / 8)
                cell_height = int((bottom_left[1] - top_left[1]) / 7)

                matrix, width, height = calculate_perspective_transform(
                    top_left, top_right, bottom_left, bottom_right, cell_width, cell_height
                )
                frame = preprocess_image(frame, 0.5)
                warped = cv2.warpPerspective(frame, matrix, (width, height))

                if warped.size != 0:
                    display_image_with_grid(warped, cell_width, cell_height)

                    rows, cols = 6, 7
                    board_state = np.zeros((rows, cols), dtype=int)

                    for i in range(rows):
                        for j in range(cols):
                            x = (1 + j) * cell_width
                            y = (1 + i) * cell_height
                            cell = warped[y:y + cell_height, x:x + cell_width]
                            color = detect_piece_color(cell)
                            board_state[i, j] = color

                    print("Board state:")
                    print(board_state)

            if output_image is not None:
                cv2.imshow("AprilTags Detected", output_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

