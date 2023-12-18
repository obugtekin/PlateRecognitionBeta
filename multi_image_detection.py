import os
import cv2
import numpy as np
from anpr import ObjectDetectionProcessor

processor = ObjectDetectionProcessor()

# Specify the path to the folder containing images
image_folder_path = 'C:/Users/Onur/Desktop/ANPR/Tensorflow/workspace/images/test'

# Iterate through images in the folder
for image_filename in os.listdir(image_folder_path):
    image_path = os.path.join(image_folder_path, image_filename)

    # Read the image
    img = cv2.imread(image_path)

    # Ensure that the image is not empty
    if img is None:
        print(f"Error reading image: {image_filename}")
        continue

    # Convert the image to the correct data type
    image_np = np.array(img)

    # Resize the image to a smaller resolution
    target_size = (800, 600)
    resized_image_np = cv2.resize(image_np, target_size)

    # Detect objects and apply OCR to the detection
    detections = processor.detect_objects(resized_image_np)
    text, region = processor.apply_ocr_to_detection(resized_image_np, detections)

# Close the image window after processing all images
cv2.destroyAllWindows()
