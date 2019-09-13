# Satellite Stereo

* Install [colmap_for_satellite_stereo](https://github.com/Kai-46/colmap_for_satellite_stereo) first.
* Use python3 instead of python2
* All the python dependent packages can be installed via

pip install -r requirements.txt

* The configuration files are inside "aoi_config/". Basically, you would need to set the "dataset_dir" and "work_dir" fields in order to get started. Inside the "steps_to_run" field, 
I've kept the necessary steps (set to true) to produce the final point cloud, with other steps (set to false) being debugging helpers.

* The directory structure for the raw dataset looks like,

{dataset_dir}/ \
&nbsp;&nbsp;&nbsp;&nbsp;    {xxx}.ntf \
&nbsp;&nbsp;&nbsp;&nbsp;    {xxx}.tar \
&nbsp;&nbsp;&nbsp;&nbsp;    {yyy}.ntf \
&nbsp;&nbsp;&nbsp;&nbsp;    {yyy}.tar \
&nbsp;&nbsp;&nbsp;&nbsp;    ...

* If you use the 3D aggregation method, the output point cloud and DSM will be inside "{work_dir}/mvs_results/aggregate_3d/"; alternatively, if the 2.5D aggregation is adopted, the output will be inside "{work_dir}/mvs_results/aggregate_2p5d/".
* Example usage: python3 stereo_pipeline.py --config_file aoi_config/aoi-d4-jacksonville.json


