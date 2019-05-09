# import matplotlib.pyplot as plt
import cv2
import numpy as np
import copy
import os


DATA_FOLDER = "data"


def resize_image(image, width=400):
    original_height, original_width, _ = image.shape
    height = int((original_height * 1.0 / original_width) * width)
    return cv2.resize(image, (width, height))


def generate_binary_image(image, threshold):
    image_clone = copy.deepcopy(image)
    image_clone[image_clone > threshold] = 255
    image_clone[image_clone <= threshold] = 0
    return image_clone

def pixel_clusters_from(binary_image):
    previous = None
    clusters = {}
    current_cluster = 0
    for p in np.argwhere(binary_image == 0):
        if previous is None:
            previous = p
            clusters[current_cluster] = []
        if abs(p[0] - previous[0]) <= 1 or abs(p[1] - previous[1]) <= 1:
            clusters[current_cluster].append(p)
            previous = p
        else:
            previous = p
            current_cluster += 1
            clusters[current_cluster] = [p]
    return clusters

if __name__ == "__main__":
    image = cv2.imread(os.path.join(DATA_FOLDER, "image_4.jpeg"))

    resized_image = resize_image(image)
    greyscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    binary_image = generate_binary_image(greyscale_image, 40)

    pixel_clusters = pixel_clusters_from(binary_image)

    for _, cluster in pixel_clusters.items():
        for pixel in cluster:
            resized_image[pixel[0], pixel[1]] = [0, 0, 255]

    cv2.imshow("Image", resized_image)
    cv2.waitKey(0)
