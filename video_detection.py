import cv2
import numpy as np
import threading
from anpr import ObjectDetectionProcessor

# Initialize the object detection processor
processor = ObjectDetectionProcessor()

# Open video capture (replace 'video_path.mp4' with the path to your video file)
cap = cv2.VideoCapture('Tensorflow/workspace/videos/sample2.mp4')

# Check if the video capture is successful
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Set the target size for resizing
target_size = (640, 480)

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

                # Display the frame with object detection and OCR result
                display_text = f"Object Detection: {text}"
                cv2.putText(resized_frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Apply Gaussian blur to the frame
                resized_frame = cv2.GaussianBlur(resized_frame, (5, 5), 0)

                # Save results if needed
                # save_results(text, region, 'video_results.csv', 'Detection_Images')
            except:
                pass

            # Display the frame with object detection
            cv2.imshow('Object Detection', resized_frame)
            cv2.waitKey(1)  # Add a small delay to let OpenCV handle events

            # Check for 'q' key to quit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

    # Release video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Start a new thread for video processing
video_thread = threading.Thread(target=process_video)
video_thread.start()

# Wait for the video thread to finish
video_thread.join()
