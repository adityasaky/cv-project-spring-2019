import cv2
import numpy as np
import copy
import os

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


def get_pixel_clusters(binary_image):
    previous = None
    clusters = {}
    current_cluster = 0
    for p in np.argwhere(binary_image == 0):
        if previous is None:
            previous = p
            clusters[current_cluster] = []
        if abs(p[0] - previous[0]) <= 1 or abs(p[1] - previous[1]) <= 1:
            clusters[current_cluster].append(p[::-1])
            previous = p
        else:
            previous = p
            current_cluster += 1
            clusters[current_cluster] = [p[::-1]]
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
                target_nn[i, j] = np.array([0, 0, 0])
            else:
                target_nn[i, j] = image[x, y]

    return target_nn


def draw_rectangle(image, center, transparency=0.4, colour=[0, 0, 255, 255], pan=[10, 10]):
    image_clone = copy.deepcopy(image)
    coordinate_1 = (center[0] - pan[0], center[1] - pan[1])
    coordinate_2 = (center[0] + pan[0], center[1] + pan[1])
    cv2.rectangle(image_clone, coordinate_1, coordinate_2, colour, thickness=-1)
    cv2.addWeighted(image_clone, transparency, image, 1 - transparency, 0, image_clone)
    return image_clone


def map_eyes(binary_image, original_image):
    original_image = copy.deepcopy(original_image)
    pixel_clusters = get_pixel_clusters(binary_image)
    cluster_centers = []
    for _, cluster in list(pixel_clusters.items())[0:2]:
        cluster_centers.append(np.add(cluster[0], cluster[len(cluster) - 1]) / 2)

    return cluster_centers


if __name__ == "__main__":
    if OUTPUT_FOLDER not in os.listdir('.'):
        os.mkdir(OUTPUT_FOLDER)

    image = cv2.imread(os.path.join(DATA_FOLDER, "image_4.jpeg"))

    resized_image = make_transparent(resize_image(image))
    greyscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    binary_image = generate_binary_image(greyscale_image, 40)
    eyes = map_eyes(binary_image, resized_image)

    transformed_image = copy.deepcopy(resized_image)
    overlaid_image = copy.deepcopy(resized_image)

    for eye in eyes:
        box_size = 10
        min_x = np.int(eye[0]) - box_size
        min_y = np.int(eye[1]) - box_size
        max_x = np.int(eye[0]) + box_size
        max_y = np.int(eye[1]) + box_size

        eye_region = transformed_image[min_x:max_x, min_y:max_y]
        ts = 3
        tx = ts * (max_x - min_x) / 4
        ty = ts * (max_y - min_y) / 4
        transformation = generate_transformation(ts, 0, -tx, 0, ts, -ty)

        transformed_image[min_x:max_x, min_y:max_y] = apply_backward_mapping(eye_region, transformation)
        overlaid_image = draw_rectangle(overlaid_image, [np.int(eye[0]), np.int(eye[1])])

    cv2.imwrite(os.path.join(OUTPUT_FOLDER, "image_4.png"), overlaid_image)
    cv2.imshow("Image", overlaid_image)

    while(1):
     key = cv2.waitKey(0)
     if key == ord('q'):
         break
