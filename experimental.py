import cv2
import numpy as np

# _make_coordinate is a helper function that allows us to sanely
# transform a (row, col) value into (x, y) simply by reversing it.
def _make_coordinate(point):
    return point[::-1]


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