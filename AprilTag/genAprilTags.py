import apriltag
import matplotlib.pyplot as plt

# Create an AprilTag detector
detector = apriltag.Detector()

# Generate and display an AprilTag
tag_id = 0
tag_family = 'tag36h11'  # Most common family
tag_size = 200  # Size of the tag in pixels

# Create a tag image
tag = detector.draw(tag_family, tag_id, tag_size)

# Save and display the tag
plt.imsave(f'apriltag_{tag_id}.png', tag, cmap='gray')
plt.imshow(tag, cmap='gray')
plt.axis('off')
plt.show()
