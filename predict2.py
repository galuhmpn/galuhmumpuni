# import necessary packages
from module import config
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
import numpy as np
import mimetypes
import argparse
import imutils
import pickle
import cv2
import os

# construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to input image or video file")
args = vars(ap.parse_args())

# determine the file type
filetype, _ = mimetypes.guess_type(args["input"])
imagePaths = [args["input"]]

# if the input is a video file, open it
if "video" in filetype:
    vs = cv2.VideoCapture(args["input"])
    imagePaths = []

    # loop over the frames from the video stream
    while True:
        grabbed, frame = vs.read()

        # break the loop if no frames left
        if not grabbed:
            break

        # create a temporary file and save the frame
        temp_path = "temp_frame.jpg"
        cv2.imwrite(temp_path, frame)
        imagePaths.append(temp_path)

# load the pre-trained model and label binarizer
print("[INFO] loading object detector...")
model = load_model(config.MODEL_PATH)
lb = pickle.loads(open(config.LB_PATH, "rb").read())

# loop over the image paths
for imagePath in imagePaths:
    # load the image and preprocess it
    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # make predictions on the image
    (boxPreds, labelPreds) = model.predict(image)
    (startX, startY, endX, endY) = boxPreds[0]

    i = np.argmax(labelPreds, axis=1)
    label = lb.classes_[i][0]

    # display the output
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]

    startX = int(startX * w)
    startY = int(startY * h)
    endX = int(endX * w)
    endY = int(endY * w)

    y = startY - 10 if startY - 10 > 10 else startY + 10
    cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

    cv2.imshow("Output", image)
    cv2.waitKey(25)

# release the video stream if applicable
if "video" in filetype:
    vs.release()
