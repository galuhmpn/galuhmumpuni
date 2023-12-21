import os

BASE_DIR = "dataset3"
IMAGES_DIR = os.path.sep.join([BASE_DIR, "images"])
ANN_DIR = os.path.sep.join([BASE_DIR,"annotations"])

BASE_OUTPUT= "output"
MODEL_DIR = os.path.sep.join([BASE_OUTPUT, "modelcoba3.h5"])
LB_DIR = os.path.sep.join([BASE_OUTPUT, "lb.pickle"])
PLOTS_DIR = os.path.sep.join([BASE_OUTPUT, "plots"])
TEST_DIR = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])

LR = 1e-4
EPOCHS = 8
BATCH_SIZE = 16