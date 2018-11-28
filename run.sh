#!/usr/bin/env bash

colmap feature_extractor --database_path ./database.db --image_path ./images/ --ImageReader.camera_model PINHOLE
colmap exhaustive_matcher --database_path ./database.db 
python3 ../remap_id.py .
colmap point_triangulator --Mapper.ba_refine_principal_point 1 --database_path ./database.db --image_path ./images/ --input_path ./init --output_path ./sparse 
colmap bundle_adjuster --input_path ./sparse --output_path ./sparse_ba \
	--BundleAdjustment.max_num_iterations 5000 \
	--BundleAdjustment.refine_principal_point 1 \
	--BundleAdjustment.function_tolerance 1e-6 \
	--BundleAdjustment.gradient_tolerance 1e-10 \
	--BundleAdjustment.parameter_tolerance 1e-8
colmap image_undistorter --max_image_size 6000 --image_path ./images --input_path ./sparse_ba --output_path ./dense
colmap patch_match_stereo --workspace_path ./dense --PatchMatchStereo.window_radius 9 --PatchMatchStereo.filter_min_triangulation_angle 1.5
colmap stereo_fusion --workspace_path ./dense --output_path ./dense/fused.ply
python3 ../geo_reference.py .
