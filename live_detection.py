import cv2
import numpy as np
import threading
from anpr import ObjectDetectionProcessor

# Initialize the object detection processor
processor = ObjectDetectionProcessor()

# Open video capture (0 for webcam)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# Set the target size for resizing
target_size = (400, 300)

# Initialize variables for skipping frames
skip_frames = 0
skip_interval = 1  # Process every 'skip_interval' frames

# Function to process video in a separate thread
def process_video():
    global skip_frames  # Declare skip_frames as global

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # Resize the frame
        resized_frame = cv2.resize(frame, target_size)

        # Skip frames based on skip_interval
        skip_frames = (skip_frames + 1) % skip_interval
        if skip_frames == 0:
            try:
                # Detect objects
                detections = processor.detect_objects(resized_frame)

                # Apply OCR to detections
                text, region = processor.apply_ocr_to_detection(resized_frame, detections)

                # Display the OCR result on the video frame
                cv2.putText(resized_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Save results if needed
                # save_results(text, region, 'realtimeresults.csv', 'Detection_Images')
            except:
                pass

            # Display the frame with object detection and OCR result
            cv2.imshow('Object Detection', resized_frame)
            key = cv2.waitKey(1)  # Add a small delay to let OpenCV handle events

            # Check for 'q' key to quit
            if key == ord('q'):
                break

    # Release video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Start a new thread for video processing
video_thread = threading.Thread(target=process_video)
video_thread.start()

# Wait for the video thread to finish
video_thread.join()
