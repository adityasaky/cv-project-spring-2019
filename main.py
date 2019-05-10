import cv2
import numpy as np
import copy
import os

DATA_FOLDER = "data"
OUTPUT_FOLDER = "output"

# _make_coordinate is a helper function that allows us to sanely
# transform a (row, col) value into (x, y) simply by reversing it.
def _make_coordinate(point):
    return point[::-1]


# resize_image scales a passed image such that it has width 400px,
# while maintaining aspect ratio.
def resize_image(image, width=400):
    original_height, original_width, _ = image.shape
    height = int((original_height * 1.0 / original_width) * width)
    return cv2.resize(image, (width, height))


# generate_binary_image thresholds a passed image at a specified value.
def generate_binary_image(image, threshold):
    image_clone = copy.deepcopy(image)
    image_clone[image_clone > threshold] = 255
    image_clone[image_clone <= threshold] = 0
    return image_clone


# get_pixel_clusters gets the clusters of pixels in order of occurrence
# top to bottom, and returns a dictionary indexed by the instance of occurrence.
def get_pixel_clusters(binary_image):
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


# make_transparent accepts an image and an alpha value, and adds an alpha
# dimension corresponding to the transparency of the image.
def make_transparent(image, alpha=255):
    b_channel, g_channel, r_channel = cv2.split(image)
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * alpha
    return cv2.merge((b_channel, g_channel, r_channel, alpha_channel))


# generate_transformation returns a 3x3 transformation matrix for passed affine
# transformation values.
def generate_transformation(a11, a12, a13, a21, a22, a23):
    return np.matrix([[a11, a12, a13],
                      [a21, a22, a23],
                      [0, 0, 1]])


# apply_backward_mapping applies nearest neighbour interpolation to display an image
# after transforming it.
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


# draw_rectangle adds an optionally transparent rectangle centered at the specified coordinate.
# The transparency parameter ranges from 0 to 1 and accepts float values.
def draw_rectangle(image, center, transparency=0.4, colour=[0, 0, 255, 255], pan=[10, 10]):
    image_clone = copy.deepcopy(image)
    # cv2.rectangle requires the coordinates in (x, y) terms and not in terms of the indices
    # of the np.matrix of the image.
    center = _make_coordinate(center)
    coordinate_1 = (center[0] - pan[0], center[1] - pan[1])
    coordinate_2 = (center[0] + pan[0], center[1] + pan[1])
    cv2.rectangle(image_clone, coordinate_1, coordinate_2, colour, thickness=-1)
    cv2.addWeighted(image_clone, transparency, image, 1 - transparency, 0, image_clone)
    return image_clone


# map_eyes identifies the centers of the eyes from clusters generated.
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
    cv2.imshow("Image", transformed_image)

    while(1):
     key = cv2.waitKey(0)
     if key == ord('q'):
         break
