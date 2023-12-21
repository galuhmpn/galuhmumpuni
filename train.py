# Import necessary packages
from settings import settings
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Flatten, Dropout, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import os

# Display information about loading the dataset
print("Status: Loading dataset...")

# Initialize lists to store data, labels, bounding boxes, and image directories
data = []
labels = []
bboxes = []
imageDirs = []

# Loop over CSV files in the annotation directory
for csvDir in paths.list_files(settings.ANN_DIR, validExts=(".csv")):
    # Load the contents of the CSV file
    rows = open(csvDir).read().strip().split("\n")
    
    # Loop over the rows in the CSV file
    for row in rows:
        row = row.split(",")
        (filename, x1, y1, x2, y2, label) = row

        # Construct the image directory
        imageDir = os.path.sep.join([settings.IMAGES_DIR, label, filename])
        
        # Read the image and get its dimensions
        image = cv2.imread(imageDir)
        (h, w) = image.shape[:2]

        # Normalize bounding box coordinates
        x1 = float(x1) / w
        y1 = float(y1) / h
        x2 = float(x2) / w
        y2 = float(y2) / h

        # Load and preprocess the image
        image = load_img(imageDir, target_size=(320, 180))  # (height, width)
        image = img_to_array(image)

        # Append the data, labels, bounding boxes, and image directories
        data.append(image)
        labels.append(label)
        bboxes.append((x1, y1, x2, y2))
        imageDirs.append(imageDir)

# Convert lists to NumPy arrays
data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)
bboxes = np.array(bboxes, dtype="float32")
imageDirs = np.array(imageDirs)

# Perform label binarization
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# Convert labels to categorical if there are two classes
if len(lb.classes_) == 2:
    labels = to_categorical(labels)

# Split the data into training and testing sets
split = train_test_split(data, labels, bboxes, imageDirs, test_size=0.20, random_state=42)
(trainImages, testImages) = split[:2]
(trainLabels, testLabels) = split[2:4]
(trainBBoxes, testBBoxes) = split[4:6]
(trainDirs, testDirs) = split[6:]

# Save test image directories to a file
print("Status: Saving test image directories...")
with open(settings.TEST_DIR, "w") as f:
    f.write("\n".join(testDirs))

# Load the MobileNetV2 model with pre-trained weights
mobilenetv2 = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(320, 180, 3)))

# Freeze the MobileNetV2 base model
mobilenetv2.trainable = False

# Extract features from MobileNetV2 model
flatten = mobilenetv2.output
flatten = Flatten()(flatten)

# Create the bounding box head
regHead = Dense(128, activation="relu")(flatten)
regHead = Dense(64, activation="relu")(regHead)
regHead = Dense(32, activation="relu")(regHead)
regHead = Dense(4, activation="sigmoid", name="bbox")(regHead)

# Create the label head for class labels
classHead = Dense(512, activation="relu")(flatten)
classHead = Dropout(0.5)(classHead)
classHead = Dense(512, activation="relu")(classHead)
classHead = Dropout(0.5)(classHead)
classHead = Dense(len(lb.classes_), activation="softmax", name="class")(classHead)

# Create the final model
model = Model(inputs=mobilenetv2.input, outputs=(regHead, classHead))

# Define loss functions and weights
losses = {"class": "categorical_crossentropy", "bbox": "MSE"}
lossWeights = {"class": 1.0, "bbox": 1.0}

# Compile the model
opt = Adam(lr=settings.LR)
model.compile(loss=losses, optimizer=opt, metrics=["accuracy"], loss_weights=lossWeights)
print(model.summary())

# Define training and testing targets
trainTargets = {"class": trainLabels, "bbox": trainBBoxes}
testTargets = {"class": testLabels, "bbox": testBBoxes}

# Train the model
print("Status: Training model...")
H = model.fit(trainImages, trainTargets, validation_data=(testImages, testTargets),
              batch_size=settings.BATCH_SIZE, epochs=settings.EPOCHS, verbose=1)

# Save the trained model
print("Status: Saving hand sign recognition model...")
model.save(settings.MODEL_DIR, save_format="h5")

# Save the label binarizer
print("Status: Saving label binarizer...")
with open(settings.LB_DIR, "wb") as f:
    f.write(pickle.dumps(lb))
