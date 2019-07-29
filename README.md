## generative_pose
Code for our ICCV 19 Paper : Monocular 3D Human Pose Estimation by Generation and Ordinal Ranking, available here : https://arxiv.org/abs/1904.01324

![Teaser Image](https://github.com/ssfootball04/generative_pose/blob/master/Teaser.png)

## Dependencies
* pytorch 
* h5py
* matplotlib

## Setup
1. Clone this repository.
2. Download the data from [Google Drive](https://drive.google.com/file/d/196RLxQKlHowEDmnJw6xLrlv16oi2z15r/view?usp=sharing) and unzip it inside the parent directory. It contains preprocessed ground truth 3D coordinates on Human3.6, and 2D pose + Ordinal Relation detections from our 2DPoseNet/OrdinalNet module. 

## Running the Code

For training MultiPoseNet, run the following command
```
python main.py --exp [name of your experiment]
```

For testing run,
```
python main.py --exp [name of your experiment] --test --numSamples [num of samples to generate] --load [pre-trained model]
```

## Pre-trained model

We provide a pre-trained model on [Google Drive](https://drive.google.com/file/d/1m6bVVms1Q54AbxrG_EE8vvzDVFVQ-JgO/view?usp=sharing). You can reproduce our results by running the test script with this model and generating 200 samples. 

## Code Layout 

This repository closely follows una_dinosauria's [Tensorflow repo](https://github.com/una-dinosauria/3d-pose-baseline) for their ICCV 17 paper A Simple Yet Effective Baseline for 3D Pose Estimation, and weigq's [Pytorch repo](https://github.com/weigq/3d_pose_baseline_pytorch) for the same. 

## Citing 

If you use this code, please cite our work : 
```
@article{sharma2019monocular,
  title={Monocular 3d human pose estimation by generation and ordinal ranking},
  author={Sharma, Saurabh and Varigonda, Pavan Teja and Bindal, Prashast and Sharma, Abhishek and Jain, Arjun},
  journal={arXiv preprint arXiv:1904.01324},
  year={2019}
}
```
