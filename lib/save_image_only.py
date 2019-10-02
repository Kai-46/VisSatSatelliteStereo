# ===============================================================================================================
#  This file is part of Creation of Operationally Realistic 3D Environment (CORE3D).                            =
#  Copyright 2019 Cornell University - All Rights Reserved                                                      =
#  -                                                                                                            =
#  NOTICE: All information contained herein is, and remains the property of General Electric Company            =
#  and its suppliers, if any. The intellectual and technical concepts contained herein are proprietary          =
#  to General Electric Company and its suppliers and may be covered by U.S. and Foreign Patents, patents        =
#  in process, and are protected by trade secret or copyright law. Dissemination of this information or         =
#  reproduction of this material is strictly forbidden unless prior written permission is obtained              =
#  from General Electric Company.                                                                               =
#  -                                                                                                            =
#  The research is based upon work supported by the Office of the Director of National Intelligence (ODNI),     =
#  Intelligence Advanced Research Projects Activity (IARPA), via DOI/IBC Contract Number D17PC00287.            =
#  The U.S. Government is authorized to reproduce and distribute copies of this work for Governmental purposes. =
# ===============================================================================================================

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio
import numpy as np


def save_image_only(matrix, save_file, cmap='magma', norm=None, save_cbar=False, plot=True):
    if not plot:
        im = matrix
        # im values should be inside [0, 1]
        eps = 1e-7
        assert (im.min() > -eps and im.max() < (1.0 + eps))
        im = np.uint8(im * 255.0)
    else:
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

        if save_cbar:
            fig_new, ax = plt.subplots()
            cbar = plt.colorbar(mpb, ax=ax, orientation='horizontal')
            ax.remove()

            # adjust color bar texts
            cbar.ax.tick_params(labelsize=10, rotation=-30)
            # save the same figure with some approximate autocropping
            fig_new.savefig(save_file + '.cbar.jpg', bbox_inches='tight')
            plt.close(fig_new)

        plt.close(fig)

    imageio.imwrite(save_file, im)
