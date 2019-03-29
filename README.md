## generative_pose
Code for our arXiv paper : Monocular 3D Human Pose Estimation by Generation and Ordinal Ranking

![Teaser Image](https://github.com/ssfootball04/generative_pose/blob/master/Teaser.png)

## Dependencies
* pytorch 
* h5py
* matplotlib

## Setup
1. Clone this repository.
2. Download the data from this url []. It contains preprocessed ground truth 3D coordinates on Human3.6, and 2D pose + Ordinal Relation detections from our 2DPoseNet/OrdinalNet module. 

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

We provide a pre-trained model at this url []. You can reproduce our results by running the test script with this model and generating 200 samples. 

## Code Layout 

This repository closely follows una_dinosauria's Tensorflow repo [] and weigq's Pytorch repo for the paper A Simple Yet Effective Baseline for 3D Pose Estimation. 

## Citing 

If you use this code, please cite our work : 
