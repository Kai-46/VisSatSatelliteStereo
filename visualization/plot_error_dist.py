import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio
import numpy as np
import cv2


def plot_err_dist(err, out_img):
    plt.figure()
    err = np.clip(err, 0, 2)
    plt.hist(err, bins='auto', cumulative=True, density=True)
    plt.xlabel('Error (meters) (clipped at 2 meters)')
    plt.ylabel('CDF')
    plt.savefig(out_img)
    plt.close()
