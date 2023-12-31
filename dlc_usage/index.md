# DeepCutLab_usage


> **DeepLabCut** is a software package for markerless pose estimation of animals performing various tasks. 

### Installation

- windows: anaconda environment;
  - install anaconda;
  - conda env create -f DLC-GPU.yaml  (deeplabcut 项目文件中)
- ubuntu: docker container;

### System Overview

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200921084842252.png)

 <p align="center">
<img src="https://static1.squarespace.com/static/57f6d51c9f74566f55ecf271/t/5cca272524a69435c3251c40/1556752170424/flowfig.jpg?format=1000w" >

```python
#main step to use deepcutlab
#导入库
import deeplabcut
#create new project
deeplabcut.create_new_project('project_name','experimenter',['path of video1','video2'])
#extract frames
deeplabcut.extract_frames(config_path)
#label frames
deeplabcut.label_frames(config_path)
#check labels(optional)
deeplabcut.check_labels(config_path)
#create dataset
deeplabcut.create_training_dataset(config_path)
#training 
deepcutlab.train_network(config_path)
#evaluate the trained net
deeplabcut.evaluate_network(config_path)
#video analyze
deeplabcut.analyze_videos(config_path,['path of videofolder'])
#plot result
deeplabcut.plot_trajectories(config_path,['path of video'])
#create a video
deeplabcut.create_labeled_video(config_path,['path of video'])
```

Directory Structure

- dlc-models:  holds the meta information with regard to the parameters of the feature detectors in cfg;
  - test;
  - train;
- labeled-data: store the frames used to create training dataset;
- training dataset: contain the training dataset and metadata about how the training dataset was created;
- videos: 

<p align="center">
<img src="https://static1.squarespace.com/static/57f6d51c9f74566f55ecf271/t/5c40f4124d7a9c0b2ce651c1/1547760716298/Box1-01.png?format=1000w" width="90%">
</p>

- mutlti-tracking: create_new_project and set flag multianimal=True;

 - [TUTORIALS:](https://www.youtube.com/channel/UC2HEbWpC_1v6i9RnDMy-dfA?view_as=subscriber) video tutorials that demonstrate various aspects of using the code base.
 - [HOW-TO-GUIDES:](/docs/UseOverviewGuide.md#how-to-guides) step-by-step user guidelines for using DeepLabCut on your own datasets (or on demo data)
 - [EXPLANATIONS:](https://github.com/DeepLabCut/DeepLabCut/wiki) resources on understanding how DeepLabCut works
 - online course [here!](https://github.com/DeepLabCut/DeepLabCut-Workshop-Materials/blob/master/summer_course2020.md)**

### ScenariesUsages

- **I have single animal videos, but I want to use the advanced tracking features & updated network capabilities introduced (for multi-animal projects) in DLC2.2:**
  - quick start: when you `create_new_project` just set the flag `multianimal=True`.

:movie_camera: [VIDEO TUTORIAL AVAILABLE!](https://youtu.be/JDsa8R5J0nQ)

Some tips: i.e. this is good for say, a hand or a mouse if you feel the "skeleton" during training would increase performance. DON'T do this for things that could be identified an individual objects. i.e., don't do whisker 1, whisker 2, whisker 3 as 3 individuals. Each whisker always has a specific spatial location, and by calling them individuals you will do WORSE than in single animal mode. 

- **I have single animal videos, but I want to use new features within in DLC2.2:**
  - quick start: when you `create_new_project` just set the flag `multianimal=Flase`, but you still get lots of upgrades! This is the typical work path for many of you. 

- **I have multiple *identical-looking animals* in my videos and I need to use DLC2.2:**
  - quick start: when you `create_new_project` set the flag `multianimal=True`. If you can't tell them apart, you can assign the "individual" ID to any animal in each frame. See this [labeling w/2.2 demo video](https://www.youtube.com/watch?v=_qbEqNKApsI)

:movie_camera: [VIDEO TUTORIAL AVAILABLE!](https://youtu.be/Kp-stcTm77g)

- **I have multiple animals, *but I can tell them apart,* in my videos and want to use DLC2.2:**
  - quick start: when you `create_new_project` set the flag `multianimal=True`. And always label the "individual" ID name the same; i.e. if you have mouse1 and mouse2 but mouse2 always has a miniscope, in every frame label mouse2 consistently. See this [labeling w/2.2 demo video](https://www.youtube.com/watch?v=_qbEqNKApsI)

:movie_camera: [VIDEO TUTORIAL AVAILABLE!](https://youtu.be/Kp-stcTm77g) - ALSO, if you can tell them apart, label animals them consistently!

- **I have a pre-2.2 single animal project, but I want to use 2.2:**
- support 2camera for 3D estimation;
- deepcutlab2.2 with extention for multitracking;

Please read [this convert 2 maDLC guide](/docs/convert_maDLC.md)

### DemoUsages

[docker](https://github.com/DeepLabCut/Docker4DeepLabCut2.0)

python

```python
import deeplabcut
deeplabcut.launch_dlc()  #open GUI
```

commandline

```shell
python -m deeplabcut
```

- https://www.youtube.com/watch?v=KcXogR-p5Ak
- https://youtu.be/Kp-stcTm77g

### Relative Paper

- DeepLabCut: markerless pose estimation of user-defined body parts with deep learning
- Using DeepLabCut for 3D markerless pose estimation across species and behaviors
- DeeperCut: A Deeper, Stronger, and Faster Multi-Person Pose Estimation Model
- Deep learning tools for the measurement of animal behavior in neuroscience

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/dlc_usage/  

