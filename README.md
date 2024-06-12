# Team12 Project : Detecting illegal dumping of garbage

### Overview
The aim of this project is to develop a computer vision system that analyzes footage recorded by surveillance cameras to detect instances where pedestrians and vehicle drivers illegally dump garbage while on the road. 

The model must be capable of performing detection across various locations and environmental conditions.

The algorithm has two main components:
- A neural network able to detect persons and garbage in a single frame 
- A simple tracker which keeps track of person and garbage identities and associates garbage with persons.

### DEMO

Person and trash detection

## Installation
The code was tested on Windows

### Setup with Conda

Create Conda environment 

```bash
conda create -n DID python=3.8.19
conda activate DID


Clone the project and go into the project directory
```bash
git clone https://github.com/TATQAQOAO/DL2024_Team12_iIlegal_dumping.git
cd DL2024_Team12_iIlegal_dumping/yolov7-main
```
Install necessary python packages using
```bash
pip install -r requirements.txt
```
Install torch and matched torchvision from [pytorch.org](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#install-linux)

The code was tested using torch 2.2.1 + cuda 11.8 and torchvision==0.17.1

Install Pytorch. You can use:

```bash
pip install torch torchvision
```
## Training
Training dataset source :

[garbage_yolov7 Image Dataset](https://universe.roboflow.com/sih-2023-ngope/garbage_yolov7)

[worker and walker Image Dataset](https://universe.roboflow.com/001-kylxv/worker-and-walker/dataset/4)

[Person Image Dataset](https://universe.roboflow.com/project-cop72/person-tylca/dataset/5)

[test Image Dataset](https://universe.roboflow.com/practice-mqbqq/test-etoky/dataset/1)

You need to download the dataset mentioned above and use the YOLOv7 format.

### Yolov7 training
Download yolov7 model and other details, please refer to [YOLOv7-Training](https://github.com/TATQAQOAO/DL2024_Team12_iIlegal_dumping/tree/main/yolov7-main)

In this project, we use [yolov7.pt](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt)
Next we can train yolov7 model
```python
python train.py --weights yolov7.pt --cfg cfg/training/yolov7_data.yaml --data data/final/data.yaml --batch-size 32 --epoch 10
```
### Detecting and tracking
For the Abandoned Garbage Detection part, we referred to [abandoned_garbage_detection](https://github.com/roym899/abandoned_bag_detection) and integrated its algorithm into detect.py.

Now we can  track the identities of people and garbage and associate the garbage with person.
```python
python detect.py --weights runs\train\exp1\weights\best.pt --source data\detection
```

## Acknowledgements
- [YOLOv7](https://github.com/wongkinyiu/yolov7)
- [abandoned_bag_detection](https://github.com/roym899/abandoned_bag_detection)
