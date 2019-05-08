# import matplotlib.pyplot as plt
import cv2
import numpy as np
import copy
import os


DATA_FOLDER = "data"


def generate_binary_image(image, threshold):
    image_clone = copy.deepcopy(image)
    image_clone[image_clone > threshold] = 255
    image_clone[image_clone <= threshold] = 0
    return image_clone


if __name__ == "__main__":
    image = cv2.cvtColor(cv2.imread(os.path.join(DATA_FOLDER, "image_3.png")), cv2.COLOR_BGR2GRAY)
    binary_image = generate_binary_image(image, 190)
    cv2.imshow("Image", binary_image)
    cv2.waitKey(0)