#  ===============================================================================================================
#  Copyright (c) 2019, Cornell University. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without modification, are permitted provided that
#  the following conditions are met:
#
#      * Redistributions of source code must retain the above copyright otice, this list of conditions and
#        the following disclaimer.
#
#      * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and
#        the following disclaimer in the documentation and/or other materials provided with the distribution.
#
#      * Neither the name of Cornell University nor the names of its contributors may be used to endorse or
#        promote products derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
#  WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE
#  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
#  TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
#  HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
#   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
#  OF SUCH DAMAGE.
#
#  Author: Kai Zhang (kz298@cornell.edu)
#
#  The research is based upon work supported by the Office of the Director of National Intelligence (ODNI),
#  Intelligence Advanced Research Projects Activity (IARPA), via DOI/IBC Contract Number D17PC00287.
#  The U.S. Government is authorized to reproduce and distribute copies of this work for Governmental purposes.
#  ===============================================================================================================


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
