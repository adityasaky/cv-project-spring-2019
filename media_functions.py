import numpy as np
import copy
import cv2


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


# make_transparent accepts an image and an alpha value, and adds an alpha
# dimension corresponding to the transparency of the image.
def make_transparent(image, alpha=255):
    if image.shape[2] != 3:
        return image
    b_channel, g_channel, r_channel = cv2.split(image)
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * alpha
    return cv2.merge((b_channel, g_channel, r_channel, alpha_channel))


# apply_backward_mapping applies nearest neighbour interpolation to display an image
# after transforming it.
def apply_backward_mapping(image, transformation):
    transformation_inverse = np.linalg.inv(transformation)
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


# probably use instead of apply_backward_mapping, delete old function
def apply_backward_mapping_eye(image, transformation, output_image):
    transformation_inverse = np.linalg.inv(transformation)
    for j in range(output_image.shape[0]):  # y value
        for i in range(output_image.shape[1]):  # x value
            target_coordinates = np.matmul(transformation_inverse, np.array([i, j, 1]).astype(float))
            (original_x, original_y, _) = image.shape
            (transformed_x, transformed_y, _) = output_image.shape
            tx = int((transformed_x - original_x))/2
            ty = int((transformed_y - original_y))/2
            x = int(np.round(target_coordinates[0, 0]) + tx)
            y = int(np.round(target_coordinates[0, 1]) + ty)
            is_out_range = x < np.floor(tx) or y < np.floor(ty) or x >= image.shape[1] + tx or y >= image.shape[0] + ty
            if not is_out_range:
                # if source pixel is black, apply transform
                test = 0
                for k in range(3):  # each b, g, r intensity
                    if image[int(y - ty), int(x - tx)][k] > 90:
                        test += 1
                if test == 0:
                    output_image[j, i] = image[int(y - ty), int(x - tx)]
    return output_image


def get_all_video_frames(file_name):
    file_path = os.path.join(DATA_FOLDER, file_name)
    video = cv2.VideoCapture(file_path)
    all_frames = []
    while video.isOpened():
        frame_check, frame = video.read()
        if frame_check:
            all_frames.append(make_transparent(frame))
        else:
            break
    video.release()
    return all_frames
