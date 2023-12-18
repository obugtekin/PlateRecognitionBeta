import os
import cv2
import numpy as np
from anpr import ObjectDetectionProcessor

processor = ObjectDetectionProcessor()

IMAGE_PATH = os.path.join('Tensorflow', 'workspace', 'images', 'test', 'Cars412.png')
img = cv2.imread(IMAGE_PATH)
image_np = np.array(img)

# Resize the image to a smaller resolution
target_size = (800, 600)
resized_image_np = cv2.resize(image_np, target_size)

detections = processor.detect_objects(resized_image_np)
text, region = processor.apply_ocr_to_detection(resized_image_np, detections)

# You can remove this line if you don't want to wait for a key press to close the image window
cv2.waitKey(0)
cv2.destroyAllWindows()
