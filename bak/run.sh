#!/usr/bin/env bash

colmap feature_extractor --database_path ./database.db \
                      --image_path ./images/ \
                      --ImageReader.camera_model PINHOLE \
                      --SiftExtraction.max_image_size 5000  \
                      --SiftExtraction.estimate_affine_shape 1 \
                      --SiftExtraction.domain_size_pooling 1

colmap exhaustive_matcher --database_path ./database.db \
                      --SiftMatching.guided_matching 1

python3 ../remap_id.py .

colmap point_triangulator --Mapper.ba_refine_principal_point 1 \
                                 --database_path ./database.db \
                                 --image_path ./images/ \
                                 --input_path ./init \
                                 --output_path ./sparse \
                                 --Mapper.filter_min_tri_angle 1 \
                                 --Mapper.tri_min_angle 1 \
                                 --Mapper.max_extra_param 1.7976931348623157e+308 \
                                 --Mapper.ba_local_max_num_iterations 40 \
                                 --Mapper.ba_local_max_refinements 3 \
                                 --Mapper.ba_global_max_num_iterations 100

colmap bundle_adjuster --input_path ./sparse --output_path ./sparse_ba \
	--BundleAdjustment.max_num_iterations 1000 \
	--BundleAdjustment.refine_principal_point 1 \
	--BundleAdjustment.function_tolerance 1e-6 \
	--BundleAdjustment.gradient_tolerance 1e-10 \
	--BundleAdjustment.parameter_tolerance 1e-8

colmap image_undistorter --max_image_size 5000 --image_path ./images --input_path ./sparse_ba --output_path ./dense

colmap patch_match_stereo --workspace_path ./dense \
                --PatchMatchStereo.window_radius 9 \
                --PatchMatchStereo.filter_min_triangulation_angle 1 \
                --PatchMatchStereo.geom_consistency 0 \
                --PatchMatchStereo.filter_min_ncc 0.05

colmap stereo_fusion --workspace_path ./dense \
                     --output_path ./dense/fused.ply \
                     --input_type photometric \
                     --StereoFusion.min_num_pixels 3

# python3 ../geo_reference.py .

