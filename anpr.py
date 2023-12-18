import os
import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import easyocr
from object_detection.utils import label_map_util, visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util

# Constants
CUSTOM_MODEL_NAME = 'my_ssd_mobnet'
LABEL_MAP_NAME = 'label_map.pbtxt'
DETECTION_THRESHOLD = 0.7
REGION_THRESHOLD = 0.2
OCR_CONFIDENCE_THRESHOLD=0.2

# Paths
ANNOTATION_PATH = os.path.join('Tensorflow', 'workspace', 'annotations')
CHECKPOINT_PATH = os.path.join('Tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME)

# Files
PIPELINE_CONFIG = os.path.join('Tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME, 'pipeline.config')
LABELMAP = os.path.join(ANNOTATION_PATH, LABEL_MAP_NAME)

class ObjectDetectionProcessor:
    def __init__(self):
        # Load trained model
        configs = config_util.get_configs_from_pipeline_file(PIPELINE_CONFIG)
        self.detection_model = model_builder.build(model_config=configs['model'], is_training=False)

        # Restore checkpoint
        ckpt = tf.train.Checkpoint(model=self.detection_model)
        ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-28')).expect_partial()

    def detect_objects(self, image_np):
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = self.detect_fn(input_tensor)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        return detections

    def detect_fn(self, image):
        image, shapes = self.detection_model.preprocess(image)
        prediction_dict = self.detection_model.predict(image, shapes)
        detections = self.detection_model.postprocess(prediction_dict, shapes)
        return detections

    def apply_ocr_to_detection(self, image_np_with_detections, detections):
        scores = list(filter(lambda x: x > DETECTION_THRESHOLD, detections['detection_scores']))
        boxes = detections['detection_boxes'][:len(scores)]

        if len(boxes) == 0:
            print("No detections found.")
            return None, None

        # Get the coordinates of the first detected box
        box = boxes[0]
        width, height = image_np_with_detections.shape[1], image_np_with_detections.shape[0]
        roi = box * [height, width, height, width]

        # Extract the region containing the car
        car_region = image_np_with_detections[int(roi[0]):int(roi[2]), int(roi[1]):int(roi[3])].copy()

        # Draw a rectangle around the detected license plate
        cv2.rectangle(image_np_with_detections, (int(roi[1]), int(roi[0])), (int(roi[3]), int(roi[2])), (0, 255, 0), 2)

        # Get the detection confidence
        confidence = scores[0]

        # Apply OCR to the license plate region
        reader = easyocr.Reader(['en', 'tr'])
        ocr_result = reader.readtext(car_region)

        # Filter OCR results based on confidence threshold
        confident_ocr_results = [result for result in ocr_result if result[2] > OCR_CONFIDENCE_THRESHOLD]

        text = self.filter_text(car_region, confident_ocr_results, REGION_THRESHOLD)

        # Display the OCR result and confidence directly on the input image
        if confident_ocr_results:
            ocr_confidence = confident_ocr_results[0][2]
            display_text = f"OCR Result: {text}, Confidence: {confidence:.2f}, OCR Confidence: {ocr_confidence:.2f}"
        else:
            display_text = f"OCR Result: {text}, Confidence: {confidence:.2f}"

        cv2.putText(image_np_with_detections, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        print(display_text)
        return text, car_region

    def filter_text(self, region, ocr_result, region_threshold):
        rectangle_size = region.shape[0] * region.shape[1]
        plate = []

        for result in ocr_result:
            length = np.sum(np.subtract(result[0][1], result[0][0]))
            height = np.sum(np.subtract(result[0][2], result[0][1]))

            if length * height / rectangle_size > region_threshold:
                plate.append(result[1])

        return plate
