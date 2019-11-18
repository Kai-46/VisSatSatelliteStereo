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


import os
from rpc_triangulate_solver.triangulate import triangulate
import numpy as np
from colmap.extract_sfm import extract_all_to_dir
from visualization.plot_reproj_err import plot_reproj_err
from debuggers.check_align import check_align
from debuggers.inspect_sfm import SparseInspector
import logging
from coordinate_system import global_to_local
import shutil


def check_sfm(work_dir, sfm_dir):
    for subdir in ['tri', 'tri_ba']:
        dir = os.path.join(sfm_dir, subdir)
        logging.info('\ninspecting {} ...'.format(dir))

        inspect_dir = os.path.join(sfm_dir, 'inspect_' + subdir)
        if os.path.exists(inspect_dir):
            shutil.rmtree(inspect_dir)

        db_path = os.path.join(sfm_dir, 'database.db')
        sfm_inspector = SparseInspector(dir, db_path, inspect_dir, camera_model='PERSPECTIVE')
        sfm_inspector.inspect_all()

        # _, xyz_file, track_file = extract_all_to_dir(dir, inspect_dir)

        # triangulate points
        # meta_file = os.path.join(work_dir, 'metas.json')
        # affine_file = os.path.join(work_dir, 'approx_camera/affine_latlonalt.json')
        # track_file = os.path.join(inspect_dir, 'kai_tracks.json')
        # out_file = os.path.join(inspect_dir, 'kai_rpc_latlonalt_coordinates.txt')
        # tmp_dir = os.path.join(inspect_dir, 'tmp')
        # triangulate(meta_file, affine_file, track_file, out_file, tmp_dir)

        # # check rpc reprojection error
        # latlonalterr = np.loadtxt(out_file)
        # plot_reproj_err(latlonalterr[:, 3], os.path.join(inspect_dir, 'rpc_reproj_err.jpg'))

        # # check alignment
        # source = np.loadtxt(xyz_file)[:, 0:3]

        # # convert lat lon alt to local
        # xx, yy, zz = global_to_local(work_dir, latlonalterr[:, 0:1], latlonalterr[:, 1:2], latlonalterr[:, 2:3])
        # target = np.hstack((xx, yy, zz))

        # np.savetxt(os.path.join(inspect_dir, 'kai_rpc_coordinates.txt'),
        #            np.hstack((target, latlonalterr[:, 3:4])),
        #            header='# format: x, y, z, reproj_err')
        # check_align(source, target)
