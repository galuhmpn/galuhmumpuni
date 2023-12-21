from module import config
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import imutils
import pickle

# Load your detection model
model = load_model(config.MODEL_PATH)
lb = pickle.loads(open(config.LB_PATH, "rb").read())

# Open the video file
cap = cv2.VideoCapture('test.mp4')

# Set the skip frames value
skip_frames = 8  # Adjust as needed

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # Break the loop if the video is finished
    if not ret:
        break

    # Skip frames
    for _ in range(skip_frames - 1):
        cap.read()

    # Preprocess the frame (resize if needed, normalize, etc.)
    frame = imutils.resize(frame, width=720)  # Adjust as needed
    preprocessed_frame = cv2.resize(frame, (224, 398)) / 255.0  # (width, height)

    # Perform object detection using your model
    (boxPreds, labelPreds) = model.predict(np.expand_dims(preprocessed_frame, axis=0))
    (startX, startY, endX, endY) = boxPreds[0]

    i = np.argmax(labelPreds, axis=1)
    label = lb.classes_[i][0]

    # Visualize the results on the frame
    startX = int(startX * frame.shape[1])
    startY = int(startY * frame.shape[0])
    endX = int(endX * frame.shape[1])
    endY = int(endY * frame.shape[0])

    y = startY - 10 if startY - 10 > 10 else startY + 10
    cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # Display the frame with detected objects
    cv2.imshow('Object Detection', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the window
cap.release()
cv2.destroyAllWindows()
