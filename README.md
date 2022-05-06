# Projector Compensation Framework using Differentiable Rendering
Author : [Jino Park](https://github.com/pjessesco), [Donghyuk Jung](https://cglab.gist.ac.kr/people/), and [Bochang Moon](https://cglab.gist.ac.kr/people/bochang.html)

Feel free to contact us by creating an issue or [email](mailto:pjessesco@gmail.com), for any question or comment.

## Prerequisite
- Our Mitsuba2 fork ([mitsuba2-pc-using-dr](https://github.com/CGLab-GIST/mitsuba2-pc-using-dr))
- Our camera application for iPad Pro ([RCDCamera](https://github.com/CGLab-GIST/RCDCamera))
- Wireless mouse for iPad (not necessary, but recommended)

## Usage

1. Install python dependencies
2. `common.py` contains global variables, methods and classes. Set `scene_path` in `common.py`. 
3. Run `RCDCamera` and set `iPadCamera` constructor in `common.py` with its IP address.
4. Run `1_capture_depth.py` in `RGBD mode`. This will capture a depth image as `exr` format.
5. Run `2_capture_color.py` in `RGB mode`. This will capture 3 color images which will used later. You need to turn on/off the light of the environment as the script guides. We recommend to use wireless mouse to control iPad to ensure static assumption of pro-cam system.
6. Set `offset_x, offset_y, transformed_width, transformed_height` in `3_dist_image_generator.py` and run it to generate target images. You may consider a color image with projection which was captured in a previous step.
7. Run `4_construct_geometry.py`. This will construct texture and geometry from captured RGB and depth images. You must check generate mesh's normal direction, it may result in unintended form.
8. Run `5_optimize_projector_pose.py`, `6_optimize_tps.py`, `7_optimize_bias_proj_img.py`.
9. For ablation study, run `8_optimize_without_warp.py`, `9_optimize_without_color_bias.py`.

## License
TODO

## Citation
```
@ARTICLE{9762256,
  author={Park, Jino and Jung, Donghyuk and Moon, Bochang},
  journal={IEEE Access}, 
  title={Projector Compensation Framework Using Differentiable Rendering}, 
  year={2022},
  volume={10},
  number={},
  pages={44461-44470},
  doi={10.1109/ACCESS.2022.3169861}}
```

## Credit
- We used python implementation of TPS by [Christoph Heindl](https://github.com/cheind/). [link](https://github.com/cheind/py-thin-plate-spline)
- We used mitsuba2 for differentiable rendering by [RGL EPFL](https://rgl.epfl.ch/). [link](https://github.com/mitsuba-renderer/mitsuba2)
- For Image viewer used in projection, we modified a code provided by [Zythyr](https://stackoverflow.com/users/4988010/zythyr) in _stack overflow_. [link](https://stackoverflow.com/a/59539843)
- For images in `reference` directory, we used dataset provided by [Bingyao Hwang](https://github.com/BingyaoHuang). [link](https://github.com/BingyaoHuang/CompenNeSt-plusplus#benchmark-dataset)
