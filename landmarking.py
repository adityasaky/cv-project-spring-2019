import numpy as np
import cv2
import math
import argparse
import enum

clicks_a, clicks_b = [], []

# image_a = cv2.resize(image_a, (200,250))
# image_b = image_b.resize(image_b, (200,250))

# image_a_copy = image_a.copy()
# image_b_copy = image_b.copy()
#
# cv2.namedWindow("image_a")
# cv2.setMouseCallback("image_a", click_and_crop_a)
# cv2.imshow("image_a", image_a_copy)
#
# cv2.namedWindow("image_b")
# cv2.setMouseCallback("image_b", click_and_crop_b)
# cv2.imshow("image_b", image_b_copy)

# key = cv2.waitKey(0)

if len(clicks_a) < 3 or len(clicks_b) < 3:
    print("Need at least 3 landmarks.")
elif len(clicks_a) != len(clicks_b):
    print("Please select corresponding landmarks alternatively. Ensure equal number of landmarks on both images.")
elif len(clicks_a) >= 3:
    print('Solving for %d pairs of landmarks'%len(clicks_a))
    x_dash = [] #np.matrix([clicks_b])
    for p in clicks_b:
        x_dash.append(p[0])
        x_dash.append(p[1])
    x_dash = np.matrix(x_dash).reshape(len(x_dash),1)
    X = []
    for p in clicks_a:
        row = [p[0], p[1], 1]
        [row.append(0) for i in range(3)]
        X.append(row)
        row = [0 for i in range(3)]
        row.extend([p[0], p[1], 1])
        X.append(row)
    X = np.matrix(X)
    a = np.linalg.pinv(X) * x_dash
    transformation_matrix = [[a[0,0], a[1,0], a[2,0]],
                            [a[3,0], a[4,0], a[5,0]],
                            [0, 0, 1]]
    # im_pil = Image.fromarray(cv2.cvtColor(image_a, cv2.COLOR_BGR2RGB))
    # applyTransformation(transformation_matrix, im_pil, Interpolation.nn)
