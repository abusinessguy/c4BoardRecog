import cv2
import apriltag
import numpy as np

def preprocess_image(image, scale_percent=50):
    # Resize the image to reduce complexity
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    # Apply Gaussian Blur to reduce noise
    blurred_image = cv2.GaussianBlur(resized_image, (3, 3), 0)

    # Apply adaptive thresholding to improve contrast
    thresholded_image = cv2.adaptiveThreshold(
        blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    return thresholded_image

def detect_apriltags(image_path, tag_family='tag36h11', scale_percent=50):
    # Load the image with AprilTags
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    # Preprocess the image
    preprocessed_image = preprocess_image(image, scale_percent)

    # Initialize the AprilTag detector options
    options = apriltag.DetectorOptions(families=tag_family)

    # Initialize the AprilTag detector
    detector = apriltag.Detector(options)

    try:
        # Detect AprilTags in the preprocessed image
        detections = detector.detect(preprocessed_image)

        if not detections:
            print("No AprilTags detected.")
        else:
            print(f"Detected {len(detections)} AprilTags.")
        
        # Convert grayscale image to BGR for colored drawing
        image_bgr = cv2.cvtColor(preprocessed_image, cv2.COLOR_GRAY2BGR)

        # Draw detections on the image
        for detection in detections:
            print(f"Detected tag ID: {detection.tag_id}")
            corners = detection.corners.astype(np.int32).reshape(-1, 1, 2)
            print(f"Corners: {corners}")

            # Draw the tag outline
            cv2.polylines(image_bgr, [corners], isClosed=True, color=(255, 0, 0), thickness=2)
            
            # Calculate the center of the tag
            center = tuple(np.mean(corners, axis=0).astype(int).flatten())
            print(f"Center: {center}")

            # Draw the tag ID at the center in blue color
            cv2.putText(image_bgr, str(detection.tag_id), center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Draw vectors from the center to each corner in green color
            for corner in corners:
                cv2.line(image_bgr, center, tuple(corner[0]), (0, 255, 0), 2)

        # Show the resized image with detected tags
        cv2.imshow('Detected AprilTags', image_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    except Exception as e:
        print(f"An error occurred during detection: {e}")

if __name__ == "__main__":
    image_path = '/projects/connectFour/TestImages/AprilTag/wooden7.jpg'
    detect_apriltags(image_path, tag_family='tag36h11', scale_percent=50)
