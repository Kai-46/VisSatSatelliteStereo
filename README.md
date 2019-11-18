# VisSat Satellite Stereo

## Introduction
This is the python interface for VISion-based SATellite stereo (VisSat) that is backed by [our adapted COLMAP](https://github.com/Kai-46/ColmapForSatelliteStereo.git). You can run both SfM and MVS on a set of satellite images.

## Installation
* Install [our adapted COLMAP](https://github.com/Kai-46/ColmapForSatelliteStereo.git) first.
* Use python3 instead of python2.
* All the python dependent packages can be installed via:

```{r, engine='bash', count_lines}
pip3 install -r requirements.txt
```

## Quick Start
* Download the [MVS3DM](https://spacenetchallenge.github.io/datasets/mvs_summary.html) satellite stereo dataset.
* The file "aoi_config/MVS3DM_Explore.json" is a template configuration for the site 'Explorer' in the MVS3DM dataset. Basically, you only need to set two fields, i.e., "dataset_dir" and "work_dir", in order to get started for this site.
* Launch our pipeline with:
```{r, engine='bash', count_lines}
python3 stereo_pipeline.py --config_file aoi_config/MVS3DM_Explorer.json
```
* If you enable "aggregate_3d", the output point cloud and DSM will be inside "{work_dir}/mvs_results/aggregate_3d/"; alternatively, if "aggregate_2p5d" is adopted, the output will be inside "{work_dir}/mvs_results/aggregate_2p5d/".
* Our pipeline is written in a module way; you can run it step by step by choosing what steps to execute in the configuration file. 
* You can navigate inside {work_dir} to get intermediate results.

## For Hackers
### General Program Logic
We use a specific directory structure to help organize the program logic. The base directory is called {work_dir}. To help understand how the system works, let me try to point out what directory or files one should pay attention to at each stage of the program. 

**SfM stage**

You need to enable {“clean_data”, “crop_image”, “derive_approx”, “choose_subset”, “colmap_sfm_perspective”} in the configuration. Then note the following files. 

1. (.ntf, .tar) pairs inside {dataset_dir}
2. (.ntf, .xml) pairs inside {work_dir}/cleaned_data
3. {work_dir}/aoi.json
4. .png inside {work_dir}/images, and .json inside {work_dir}/metas
5. .json inside {work_dir}/approx_camera, especially perspective_enu.json
6. {work_dir}/colmap/subset_for_sfm/{images, perspective_dict.json}
7. {work_dir}/colmap/sfm_perspective/init_ba_camera_dict.json

Step [1-4] transform the (.ntf, .tar) data into more accessible conventional formats. Step 5 approximates the RPC cameras with perspective cameras. Step [6-7] selects a subset of images (by default, all the images), performs bundle adjustment, and writes bundle-adjusted camera parameters to {work_dir}/colmap/sfm_perspective/init_ba_camera_dict.json. 
For perspective cameras in the .json files mentioned in Step [5-7], the camera parameters are organized as:
```{r, engine='bash'}
w, h, f_x, f_y, c_x, c_y, s, q_w, q_x, q_y, q_z, t_x, t_y, t_z
```
, where (w,h) is image size, (f_{x,y}, c_{x,y}, s) are camera intrinsics, q_{w,x,y,z} is the quaternion representation of the rotation matrix, and t_{x,y,z}is the translation vector.


**Coordinate system**

Our perspective cameras use the local ENU coordinate system instead of the global (lat, lon, alt) or (utm east, utm north, alt). 

For conversion between (lat, lon, alt) and local ENU, please refer to: coordinate_system.py and latlonalt_enu_converter.py

For conversion between (lat, lon) and (utm east, utm north), please refer to: lib/latlon_utm_converter.py


**MVS stage**

To run MVS after the SfM stage is done, you need to enable {“reparam_depth”, “colmap_mvs”, “aggregate_3d”} or {“reparam_depth”, “colmap_mvs”, “aggregate_2p5d”}.


