This is a fork of [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn). See the original repository for details.

# Overview
This repository implements a neural model to encode material properties on the surfaces of 3D objects as described in the thesis _Neural Compression of Material Properties using a Geometry-Associated Feature Hierarchy_. 
It builds on [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn), a fast framework to train and evaluate small MLPs.
We contribute a novel input encoding as described in the thesis as well as an example.

# Usage
```
neural_surface <path_to_object_basedir> <object_filename> <sample_path>
```
- `path_to_object_basedir` is the filepath to the directory containing the .obj, .mtl and texture files
- `object_filename` is the filename of the .obj file in the base directory
- `sample_path` is the path to a .csv file containing a list of surface positions to validate the model

The validation file is structured as follows:
```
image_width, image_height 		// Width and height of the validation image
triangle_id, t_0, t_1			// Each row represents a pixel in the image and contains the surface position
triangle_id, t_0, t_1			// 		 that the pixel lands on (t_2 is implicit)
...
```
