# Satellite Stereo

## Intro
This is the python interface for VISion-based SATellite stereo (VisSat) that is backed by [our adapted COLMAP](https://github.com/Kai-46/ColmapForSatelliteStereo.git). You can run both SfM and MVS on a set of satellite images.

## Installation
* Install [our adapted COLMAP](https://github.com/Kai-46/ColmapForSatelliteStereo.git) first.
* Use python3 instead of python2
* All the python dependent packages can be installed via

```{r, engine='bash', count_lines}
pip3 install -r requirements.txt
```

## Quick Usage
* Download the [MVS3DM](https://spacenetchallenge.github.io/datasets/mvs_summary.html) satellite stereo dataset.
* The file "aoi_config/MVS3DM_Explore.json" is an template configuration for the site 'Explorer' in the MVS3DM dataset. Basically, you only need to set two fields, i.e., "dataset_dir" and "work_dir", in order to get started. 
* Then, launch our pipeline with:
```{r, engine='bash', count_lines}
python3 stereo_pipeline.py --config_file aoi_config/MVS3DM_Explorer.json
```
* If you use the 3D aggregation method, the output point cloud and DSM will be inside "{work_dir}/mvs_results/aggregate_3d/"; alternatively, if the 2.5D aggregation is adopted, the output will be inside "{work_dir}/mvs_results/aggregate_2p5d/".
* Our pipeline is written in a module way; you can run it step by step by choosing what steps to execute in the configuration file. 
* You can also navigate inside {work_dir} to get intermediate results.



