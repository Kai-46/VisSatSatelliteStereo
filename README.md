# Satellite Stereo

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

* The output point cloud would be inside "{work_dir}/mvs_results/aggregate_2p5d/aggregate_2p5d.ply".


