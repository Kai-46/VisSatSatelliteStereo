import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio
import numpy as np
import cv2


def save_image_only(matrix, save_file, maskout=None, cmap='magma', norm=None, save_cbar=False, save_mask=False, plot=True):
    if not plot:
        im = matrix
        # im values should be inside [0, 1]

        nan_mask = np.any(np.isnan(matrix), axis=2)
        # for visualization
        matrix[nan_mask] = 0.0

        if maskout is not None:
            nan_mask = np.logical_or(nan_mask, maskout)
        nan_mask = np.tile(nan_mask[:, :, np.newaxis], (1, 1, 3))

        eps = 1e-7
        assert (im.min() > -eps and im.max() < (1.0 + eps))
        im = np.uint8(im * 255.0)
    else:
        # for visualization, set nan value to nanmin
        nan_mask = np.isnan(matrix)
        matrix[nan_mask] = np.nanmin(matrix)

        if maskout is not None:
            nan_mask = np.logical_or(nan_mask, maskout)
        # add third channel
        nan_mask = np.tile(nan_mask[:, :, np.newaxis], (1, 1, 3))

        fig = plt.figure()
        dpi = fig.get_dpi()
        fig.set_size_inches(matrix.shape[1] / dpi, matrix.shape[0] / dpi)

        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.axis('off')
        fig.add_axes(ax)

        if norm is None:
            mpb = ax.imshow(matrix, cmap=cmap)
        else:
            mpb = ax.imshow(matrix, cmap=cmap, norm=norm)

        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        im = data.reshape((h, w, 3)).astype(dtype=np.uint8)

        # resize (im size might mismatch matrix size by 1)
        im = cv2.resize(im, (matrix.shape[1], matrix.shape[0]), interpolation=cv2.INTER_NEAREST)

        if save_cbar:
            fig_new, ax = plt.subplots()
            cbar = plt.colorbar(mpb, ax=ax, orientation='horizontal')
            ax.remove()

            # adjust color bar texts
            cbar.ax.tick_params(labelsize=10, rotation=-30)
            # save the same figure with some approximate autocropping
            idx = save_file.rfind('.')
            fig_new.savefig(save_file[:idx] + '.cbar.jpg', bbox_inches='tight')
            plt.close(fig_new)

        plt.close(fig)

    im[nan_mask] = 0
    imageio.imwrite(save_file, im)

    if save_mask:
        idx = save_file.rfind('.')
        valid_mask = 1.0 - np.float32(nan_mask[:, :, 0])
        imageio.imwrite(save_file[:idx] + '.mask.jpg', np.uint8(valid_mask * 255.0))
