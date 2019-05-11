import cv2
import numpy as np
import copy
import os
from landmarking import solve_landmarks

DATA_FOLDER = "data"
OUTPUT_FOLDER = "output"

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


def make_transparent(image, alpha=255):
    b_channel, g_channel, r_channel = cv2.split(image)
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * alpha
    return cv2.merge((b_channel, g_channel, r_channel, alpha_channel))

def generate_transformation(a11, a12, a13, a21, a22, a23):
    return np.matrix([[a11, a12, a13],
                      [a21, a22, a23],
                      [0, 0, 1]])

def apply_backward_mapping(image, transformation):
    transformation_inverse = np.linalg.pinv(transformation)
    target_nn = copy.deepcopy(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            target_coordinates = np.dot(transformation_inverse, np.transpose(np.matrix([[i, j, 1]])))

            x = np.int32(np.round(target_coordinates[0, 0]))
            y = np.int32(np.round(target_coordinates[1, 0]))
            if x < 0 or y < 0 or x >= image.shape[0] or y >= image.shape[1]:
                target_nn[i, j] = np.array([0, 0, 0, 0])
            else:
                target_nn[i, j] = image[x, y]

    return target_nn

def map_eyes(binary_image, original_image):
    original_image = copy.deepcopy(original_image)
    pixel_clusters = pixel_clusters_from(binary_image)
    cluster_centers = []
    for _, cluster in list(pixel_clusters.items())[0:2]:
        cluster_centers.append(np.add(cluster[0], cluster[len(cluster) - 1]) / 2)

    # x_pan = 10
    # y_pan = 10
    #
    # for pixel in cluster_centers:
    #     center_x = np.int(pixel[0])
    #     center_y = np.int(pixel[1])
    #     for x in range(center_x - x_pan, center_x + x_pan + 1):
    #         for y in range(center_y - y_pan, center_y + y_pan + 1):
    #             original_image[x, y] = [0, 0, 255, 1]
    return cluster_centers

if __name__ == "__main__":
    if OUTPUT_FOLDER not in os.listdir('.'):
        os.mkdir(OUTPUT_FOLDER)

    image = cv2.imread(os.path.join(DATA_FOLDER, "image_4.jpeg"))

    resized_image = make_transparent(resize_image(image))
    greyscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    binary_image = generate_binary_image(greyscale_image, 40)
    eyes = map_eyes(binary_image, resized_image)

    # cv2.imwrite(os.path.join(OUTPUT_FOLDER, "red_birb.png"), resized_image)

    transformed_image = copy.deepcopy(resized_image)

    # for eye in eyes:
    #     box_size = 15
    #     min_x = np.int(eye[0]) - box_size
    #     min_y = np.int(eye[1]) - box_size
    #     max_x = np.int(eye[0]) + box_size
    #     max_y = np.int(eye[1]) + box_size
    #
    #     eye_region = transformed_image[min_x:max_x, min_y:max_y]
    #     ts = 3
    #     tx = ts * (max_x - min_x) / 4
    #     ty = ts * (max_y - min_y) / 4
    #     transformation = generate_transformation(ts, 0, -tx, 0, ts, -ty)
    #
    #     transformed_image[min_x:max_x, min_y:max_y] = apply_backward_mapping(eye_region, transformation)

    eye_width = np.linalg.norm(eyes[0] - eyes[1])
    eye_width = eye_width if eye_width > 40 else 100
    clipart1 = cv2.imread(os.path.join(DATA_FOLDER, "clipart1.png"), cv2.IMREAD_UNCHANGED)
    clipart1 = resize_image(clipart1, np.int(eye_width*1.5))

    clipart_eyemark_y = clipart1.shape[0] / 2
    clipart_eyemark_x = [np.int(clipart1.shape[1] * 0.25), np.int(clipart1.shape[0] * 0.75)]

    # clipart_landmarks = [(clipart_eyemark_y, clipart_eyemark_x[0]), (clipart_eyemark_y, (clipart_eyemark_x[0] + clipart_eyemark_x[1])/2), (clipart_eyemark_y, clipart_eyemark_x[1])]
    # target_landmarks = [eyes[0], np.add(eyes[0], eyes[1])/2, eyes[1]]

    # clipart_transformation = solve_landmarks(clipart_landmarks, target_landmarks)
    # clipart1 = apply_backward_mapping(clipart1, clipart_transformation)

    min_y, min_x = np.subtract(eyes[1], (clipart1.shape[0] / 2, clipart1.shape[1] / 6))
    min_x, min_y = np.int(min_x), np.int(min_y)

    # max_x, max_y = np.add((min_x, min_y), (clipart1.shape[0], clipart1.shape[1]))
    # max_x, max_y = np.int(max_x), np.int(max_y)

    # transformed_image[min_x:max_x, min_y:max_y] = clipart1
    for x in range(0, clipart1.shape[1]):
        for y in range(0, clipart1.shape[0]):
            # transformed_image[min_y + y, min_x + x] = [0,0,255,1]
            if clipart1[y,x][3] > 125:
                transformed_image[min_y + y, min_x + x] = clipart1[y, x]


    cv2.imwrite(os.path.join(OUTPUT_FOLDER, "cliparted_image_temp1.png"), transformed_image)
