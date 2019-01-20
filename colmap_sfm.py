import colmap_sfm_helper
from lib.run_cmd import run_cmd


def run_sfm(sfm_dir, camera_model):
    assert(camera_model == 'PINHOLE' or camera_model == 'PERSPECTIVE')

    # feature extraction
    cmd = 'colmap feature_extractor --database_path {sfm_dir}/database.db \
                                    --image_path {sfm_dir}/images/ \
                                    --ImageReader.camera_model {camera_model} \
                                    --SiftExtraction.max_image_size 5000  \
                                    --SiftExtraction.estimate_affine_shape 1 \
                                    --SiftExtraction.domain_size_pooling 1'.format(sfm_dir=sfm_dir, camera_model=camera_model)
    run_cmd(cmd)

    # seems that we need to copy camera intrinsics into database
    # this is important??  Maybe not

    # feature matching
    cmd = 'colmap exhaustive_matcher --database_path {sfm_dir}/database.db \
                                            --SiftMatching.guided_matching 1'.format(sfm_dir=sfm_dir)

    run_cmd(cmd)

    # create initial poses
    colmap_sfm_helper.create_init_files(sfm_dir)

    # triangulate points
    cmd = 'colmap point_triangulator --Mapper.ba_refine_principal_point 1 \
                                             --database_path {sfm_dir}/database.db \
                                             --image_path {sfm_dir}/images/ \
                                             --input_path {sfm_dir}/init \
                                             --output_path {sfm_dir}/sparse \
                                             --Mapper.filter_min_tri_angle 24.999 \
                                             --Mapper.tri_min_angle 25 \
                                             --Mapper.filter_max_reproj_error 3 \
                                             --Mapper.max_extra_param 1e100 \
                                             --Mapper.ba_local_num_images 6 \
                                             --Mapper.ba_local_max_num_iterations 100 \
                                             --Mapper.ba_global_images_ratio 1.0000001\
                                             --Mapper.ba_global_max_num_iterations 100'.format(sfm_dir=sfm_dir)
    run_cmd(cmd)

    # normalize sparse reconstruction
    cmd = 'colmap normalize --input_path {sfm_dir}/sparse --output_path {sfm_dir}/sparse_ba'.format(sfm_dir=sfm_dir)
    run_cmd(cmd)

    # global bunble adjustment
    cmd = 'colmap bundle_adjuster --input_path {sfm_dir}/sparse_ba --output_path {sfm_dir}/sparse_ba \
    	                                    --BundleAdjustment.max_num_iterations 5000 \
    	                                    --BundleAdjustment.refine_principal_point 1 \
    	                                    --BundleAdjustment.function_tolerance 1e-6 \
    	                                    --BundleAdjustment.gradient_tolerance 1e-8 \
    	                                    --BundleAdjustment.parameter_tolerance 1e-8'.format(sfm_dir=sfm_dir)
    run_cmd(cmd)

