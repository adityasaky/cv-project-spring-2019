# import matplotlib.pyplot as plt
import cv2
import numpy as np
import copy
import os


DATA_FOLDER = "data"


def resize_image(image, width=400):
    original_height, original_width = image.shape
    height = int((original_height * 1.0 / original_width) * width)
    return cv2.resize(image, (width, height))


def generate_binary_image(image, threshold):
    image_clone = copy.deepcopy(image)
    image_clone[image_clone > threshold] = 255
    image_clone[image_clone <= threshold] = 0
    return image_clone


if __name__ == "__main__":
    image = cv2.cvtColor(cv2.imread(os.path.join(DATA_FOLDER, "image_4.jpeg")), cv2.COLOR_BGR2GRAY)

    resized_image = resize_image(image)
    binary_image = generate_binary_image(resized_image, 40)

    cv2.imshow("Image", binary_image)
    cv2.waitKey(0)