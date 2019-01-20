import numpy as np


def vector_angle(vec1, vec2):
    vec1 = vec1 / np.sqrt(np.dot(vec1.T, vec1))
    vec2 = vec2 / np.sqrt(np.dot(vec2.T, vec2))

    tmp = np.dot(vec1.T, vec2)
    angle = np.rad2deg(np.math.acos(tmp))

    return angle