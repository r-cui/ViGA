# ViGA: Video moment retrieval via Glance Annotation
This is the official repository of the paper "Video Moment Retrieval from Text Queries via Single Frame Annotation" published in SIGIR 2022.

https://arxiv.org/abs/2204.09409

##  Dependencies
This project has been tested on the following conda environment.
```
$ conda create --name viga python=3.7
$ source activate viga
(viga)$ conda install pytorch=1.10.0 cudatoolkit=11.3.1
(viga)$ pip install numpy scipy pyyaml tqdm 
```

##  Data preparation
This repository contains our glance annotations already. To replicate our work, one should prepare extra data and finally get the following structure.
```
ckpt/                                 our pre-trained model, available at https://drive.google.com/file/d/1S4e8XmIpiVFJKSSJ4Tig4qN0yaCwiVLs/view?usp=sharing
data/
+-- activitynetcaptions/
|   +-- c3d/                    
|   +-- annotations/
|   |   +-- glance/
|   |   |   +-- train.json                
|   |   |   +-- val_1.json                
|   |   |   +-- val_2.json   
|   |   +-- train.json                downloaded
|   |   +-- val_1.json                downloaded
|   |   +-- val_2.json                downloaded
+-- charadessta/
|   +-- i3d/                     
|   +-- c3d/ 
|   +-- vgg/
|   +-- annotations/
|   |   +-- glance/
|   |   |   +-- charades_sta_train.txt
|   |   |   +-- charades_sta_test.txt
|   |   +-- charades_sta_train.txt    downloaded
|   |   +-- charades_sta_test.txt     downloaded
|   |   +-- Charades_v1_train.csv     downloaded
|   |   +-- Charades_v1_test.csv      downloaded
+-- tacos/
|   +-- c3d/ 
|   +-- annotations/
|   |   +-- glance/
|   |   |   +-- train.json                
|   |   |   +-- test.json                 
|   |   |   +-- val.json
|   |   +-- train.json                downloaded
|   |   +-- test.json                 downloaded
|   |   +-- val.json                  downloaded
glove.840B.300d.txt                   downloaded from https://nlp.stanford.edu/data/glove.840B.300d.zip
```

### 1. ActivityNet Captions
#### c3d feature
Downloaded from http://activity-net.org/challenges/2016/download.html. We extracted the features from `sub_activitynet_v1-3.c3d.hdf5` as individual files.

Folder contains 19994 `vid.npy`s, each of shape (T, 500).

#### annotation
Downloaded from https://cs.stanford.edu/people/ranjaykrishna/densevid/

### 2. Charades-STA
#### c3d feature 

We extracted the C3D features of Charades-STA by ourselves. We decided not to make it available for downloading due to our limited resource of online storage, and the fact that this process can be easily replicated. To specify, we directly adopted the C3D model weights pre-trained on Sports1M. The extracted feature of each clip in a video was the `fc6` layer, and the clips were sampled via sliding window of `step size of 8, window size of 16`. Our codes for this extraction were based on this repo. https://github.com/DavideA/c3d-pytorch

Folder contains 9848 `vid.npy`s, each of shape (T, 4096).


#### i3d feature 
Downloaded from https://github.com/JonghwanMun/LGI4temporalgrounding. This is the features extracted from I3D (finetuned on Charades). We processed them by trimming off unnecessary dimensions.

Folder contains 9848 `vid.npy`s, each of shape (T, 1024).

#### vgg feature 
Downloaded from https://github.com/microsoft/2D-TAN. We processed the data by converting the downloaded version `vgg_rgb_features.hdf5` into numpy arrays.

Folder contains 6672 `vid.npy`s, each of shape (T, 4096).
#### annotation
Downloaded from https://github.com/jiyanggao/TALL

### 3. TACoS
#### c3dfeature 
Downloaded from https://github.com/microsoft/2D-TAN. We extracted the features from `tall_c3d_features.hdf5` as individual files.

Folder contains 127 `vid.npy`s, each of shape (T, 4096).
#### annotation
Downloaded from https://github.com/microsoft/2D-TAN
## Run
Our models were trained using the following commands.
```
(viga)$ CUDA_VISIBLE_DEVICES=0 python -m src.experiment.train --task activitynetcaptions
(viga)$ CUDA_VISIBLE_DEVICES=0 python -m src.experiment.train --task charadessta
(viga)$ CUDA_VISIBLE_DEVICES=0 python -m src.experiment.train --task tacos
```
Our trained models were evaluated using the following commands.
```
(viga)$ CUDA_VISIBLE_DEVICES=0 python -m src.experiment.eval --exp ckpt/activitynetcaptions
(viga)$ CUDA_VISIBLE_DEVICES=0 python -m src.experiment.eval --exp ckpt/charadessta_c3d
(viga)$ CUDA_VISIBLE_DEVICES=0 python -m src.experiment.eval --exp ckpt/charadessta_i3d
(viga)$ CUDA_VISIBLE_DEVICES=0 python -m src.experiment.eval --exp ckpt/charadessta_vgg
(viga)$ CUDA_VISIBLE_DEVICES=0 python -m src.experiment.eval --exp ckpt/tacos
```
