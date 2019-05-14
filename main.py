import cv2
import numpy as np
import copy
import os
from landmarking import solve_landmarks
from media_functions import resize_image, generate_binary_image, make_transparent, apply_backward_mapping, apply_backward_mapping_eye, get_all_video_frames
from constants import DATA_FOLDER, CLIPART_FOLDER, OUTPUT_FOLDER


def in_cluster_neighbor(pixel, cluster):
    for member in cluster[::-1]:
        if abs(pixel[0] - member[0]) <= 2 and abs(pixel[1] - member[1]) <= 2:
            return True
    return False

# get_pixel_clusters gets the clusters of pixels in order of occurrence
# top to bottom, and returns a dictionary indexed by the instance of occurrence.
def get_pixel_clusters(binary_image, merge_clusters = False):
    clusters = {}
    current_cluster = 0
    for p in np.argwhere(binary_image == 0):
        if current_cluster not in clusters:
            clusters[current_cluster] = [p]
            continue
        if in_cluster_neighbor(p, clusters[current_cluster]):
            clusters[current_cluster].append(p)
        else:
            current_cluster += 1
            clusters[current_cluster] = [p]
    if merge_clusters:
        clusters = merge_connected_clusters(clusters)
    return clusters

def merge_connected_clusters(clusters):
    current = 0
    while current < len(clusters):
        comparison = current + 1
        comparisonBreak = False
        while not comparisonBreak:
            if comparison >= len(clusters):
                break
            for pixel in clusters[current]:
                if in_cluster_neighbor(pixel, clusters[comparison]):
                    clusters[comparison].extend(clusters[current])
                    clusters.pop(current)
                    clusters = serialize_clusters(clusters)
                    current = -1
                    comparisonBreak = True
                    break
            comparison += 1
        current += 1
    return clusters

def serialize_clusters(clusters):
    index = 0
    result = {}
    for cluster in clusters:
        result[index] = clusters[cluster]
        index += 1
    return result

# generate_transformation returns a 3x3 transformation matrix for passed affine
# transformation values.
def generate_transformation(a11, a12, a13, a21, a22, a23):
    return np.matrix([[a11, a12, a13],
                      [a21, a22, a23],
                      [0, 0, 1]])


# map_eyes identifies the cluster min/max coordinates of the eyes from clusters generated.
def map_eyes(binary_image, original_image):
    pixel_clusters = get_pixel_clusters(binary_image)
    cluster_centers = []
    for _, cluster in list(pixel_clusters.items()):
        cluster_centers.append(np.add(cluster[0], cluster[len(cluster) - 1]) / 2)

    cluster_centers_two = sorted(cluster_centers[:2], key=lambda i: i[1])
    if len(cluster_centers_two) < 2 or abs(cluster_centers_two[0][0] - cluster_centers_two[1][0]) > 40:
        cluster_centers_two = sorted(cluster_centers_two, key=lambda i: i[0])
        if len(cluster_centers_two) > 1:
            cluster_centers_two.pop()
        cluster_center = cluster_centers_two[0]
        center_col = cluster_center[1]
        left_count = 0
        right_count = 0
        for center in cluster_centers:
            if center[0] >= original_image.shape[0] / 2:
                continue
            if np.array_equal(center, cluster_center):
                continue
            if center[1] > center_col:
                right_count += 1
            elif center[1] < center_col:
                left_count += 1
            else:
                if center_col <= original_image.shape[1]:
                    right_count += 1
                else:
                    left_count += 1
        if left_count == right_count:
            if center_col <= original_image.shape[1] / 2:
                # this is the right eye
                return cluster_centers_two, 'right'
            else:
                return cluster_centers_two, 'left'
        elif left_count < right_count:
            return cluster_centers_two, 'right'
        else:
            return cluster_centers_two, 'left'
    return cluster_centers_two, None


def map_eyes_bounds(binary_image):
    pixel_clusters = get_pixel_clusters(binary_image, False)
    print(pixel_clusters)
    cluster_dimensions = []  # 2 DEFAULT FOR NOW?

    # MODIFY TO ONLY TAKE ONE EYE IF NEEDED? (current modification: eye distance at 45 pixels apart)
    for cluster in list(pixel_clusters.values())[0:2]:
        cluster = np.asarray(cluster)
        min_xy = np.min(cluster, axis=0)
        max_xy = np.max(cluster, axis=0)
        cluster_dimensions.append([min_xy[0], max_xy[0], min_xy[1], max_xy[1]]) # y-coord, x-coord
    # sort by lowest y value min, to get topmost clusters
    cluster_dimensions = sorted(cluster_dimensions, key=lambda i: i[0])

    # if np.linalg.norm(cluster_centers[0][1][1] - cluster_centers[1][1][1]) > 45:
    #     cluster_centers.pop()

    print(cluster_dimensions)
    return cluster_dimensions


def clipart_on_eyes(image, eyes, clipart, position=None, meta = (0,0)):
    eye_width = np.linalg.norm(eyes[0] - eyes[1]) if len(eyes) == 2 else 100
    eye_width = eye_width if eye_width > 40 else 100

    clipart1 = resize_image(copy.deepcopy(clipart), np.int(eye_width*1.5))
    image = copy.deepcopy(image)
    # clipart_eyemark_y = clipart1.shape[0] / 2
    # clipart_eyemark_x = [np.int(clipart1.shape[1] * 0.25), np.int(clipart1.shape[0] * 0.75)]

    # clipart_landmarks = [(clipart_eyemark_y, clipart_eyemark_x[0]), (clipart_eyemark_y, (clipart_eyemark_x[0] + clipart_eyemark_x[1])/2), (clipart_eyemark_y, clipart_eyemark_x[1])]
    # target_landmarks = [eyes[0], np.add(eyes[0], eyes[1])/2, eyes[1]]
    # clipart_transformation = solve_landmarks(clipart_landmarks, target_landmarks)
    # clipart1 = apply_backward_mapping(clipart1, clipart_transformation)

    min_y, min_x = np.subtract(eyes[0], (clipart1.shape[0] / 2, clipart1.shape[1] / 6))
    min_x, min_y = np.int(min_x + meta[0]), np.int(min_y + meta[1])
    if position == 'right':
        min_x -= 100

    # max_x, max_y = np.add((min_x, min_y), (clipart1.shape[0], clipart1.shape[1]))
    # max_x, max_y = np.int(max_x), np.int(max_y)

    # transformed_image[min_x:max_x, min_y:max_y] = clipart1
    for x in range(0, clipart1.shape[1]):
        for y in range(0, clipart1.shape[0]):
            # transformed_image[min_y + y, min_x + x] = [0,0,255,1]
            if clipart1[y,x][3] > 100 and 0 <= (min_y + y) < image.shape[0] and 0 <= (min_x + x) < image.shape[1]:
                image[min_y + y, min_x + x] = clipart1[y, x][0:3]

    # cv2.imwrite(os.path.join(OUTPUT_FOLDER, "cliparted_image_temp1.png"), transformed_image)
    # print(eyes, (min_y, min_x), eye_width)
    return image


if __name__ == "__main__":
    if OUTPUT_FOLDER not in os.listdir('.'):
        os.mkdir(OUTPUT_FOLDER)

    image = cv2.imread(os.path.join(DATA_FOLDER, "test_clusters.jpg"))

    resized_image = resize_image(image)
    greyscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    binary_image = generate_binary_image(greyscale_image, 45)
    eye_dimensions = map_eyes_bounds(binary_image)

    transformed_image = copy.deepcopy(resized_image)
    overlaid_image = copy.deepcopy(resized_image)

    ts = 2

    for i in range(len(eye_dimensions)):

        # if box size is small, manually make it a little larger
        if eye_dimensions[i][1] - eye_dimensions[i][0] < 2 or eye_dimensions[i][3] - eye_dimensions[i][2] < 2:
            eye_dimensions[i][0] -= 1
            eye_dimensions[i][1] += 1
            eye_dimensions[i][2] -= 1
            eye_dimensions[i][3] += 1

        # get centers of both cropped source and transformed images (will be the same)
        center_y = int((eye_dimensions[i][0] + eye_dimensions[i][1]) / 2)
        center_x = int((eye_dimensions[i][2] + eye_dimensions[i][2]) / 2)

        # calculate range of scaled transformed image; why does it look better with + 1?
        # (ts = 3 and no +1 looks messed up)
        range_y = ts * (int(eye_dimensions[i][1]) - int(eye_dimensions[i][0]) + 1)
        range_x = ts * (int(eye_dimensions[i][3]) - int(eye_dimensions[i][2]) + 1)

        # calculate min/max coordinate values of transformed image
        min_y = center_y - np.int(range_y/2)
        max_y = center_y + np.int(range_y/2)
        min_x = center_x - np.int(range_x/2)
        max_x = center_x + np.int(range_x/2)

        # get source and transformed images (pre-transform)
        eye_region = transformed_image[eye_dimensions[i][0]:eye_dimensions[i][1] + 1,
                     eye_dimensions[i][2]:int(eye_dimensions[i][3]) + 1]
        output_image = transformed_image[min_y:max_y + 1, min_x:max_x + 1]

        # no translation - that is applied through indices later
        # consider how translation up and left would not work when going forward
        transformation = generate_transformation(ts, 0, 0, 0, ts, 0)

        # transformed_image[min_y:max_y, min_x:max_x] = \
        #     apply_backward_mapping_eye(eye_region, eye_dimensions[i][0] - min_y, eye_dimensions[i][2] - min_x,
        #                                transformation, output_image)
        transformed_image[min_y:max_y + 1, min_x:max_x + 1] = \
            apply_backward_mapping_eye(eye_region, transformation, output_image)
        #overlaid_image = draw_rectangle(overlaid_image, [np.int(eye[0]), np.int(eye[1])])

    cv2.imwrite(os.path.join(OUTPUT_FOLDER, "image_1.png"), overlaid_image)
    # cv2.imshow("Image", transformed_image)
    # cv2.waitKey()

    clipart_name = "hat"
    clipart = cv2.imread(os.path.join(CLIPART_FOLDER, clipart_name + ".png"), cv2.IMREAD_UNCHANGED)
    meta_reader = open(os.path.join(CLIPART_FOLDER, clipart_name + ".meta"), "r")
    if meta_reader:
        clipart_meta = (int(meta_reader.readline()), int(meta_reader.readline()))
    else:
        clipart_meta = (0,0)
    meta_reader.close()

    file_path = os.path.join(DATA_FOLDER, "video_2_compressed.mov")
    video_reader = cv2.VideoCapture(file_path)

    video_size = (int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    codec = cv2.VideoWriter_fourcc(*'jpeg')
    video_writer = cv2.VideoWriter(os.path.join(OUTPUT_FOLDER, "outpy2.mov"), codec, 16, video_size)

    all_frames = []
    index = -1
    while video_reader.isOpened():
        index += 1
        print("processing frame: ", index)
        frame_check, frame = video_reader.read()
        if frame_check:
            greyscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            binary_frame = generate_binary_image(greyscale_frame, 40)

            eyes, position = map_eyes(binary_frame, frame)
            # cv2.imshow("Frame", binary_frame)
            # cv2.waitKey(0)
            processed_frame = clipart_on_eyes(frame, eyes, clipart, position, clipart_meta)
            video_writer.write(processed_frame)
        else:
            break

    video_reader.release()
    video_writer.release()
