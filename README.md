# Team12 Project : Detecting illegal dumping of garbage

### Overview
The aim of this project is to develop a computer vision system that analyzes footage recorded by surveillance cameras to detect instances where pedestrians and vehicle drivers illegally dump garbage while on the road. 

The model must be capable of performing detection across various locations and environmental conditions.

The algorithm has two main components:
- 
- 

### DEMO


Person and trash detection

## Installation
The code was tested on 


### Setup with Conda

Create Conda environment 

```bash
conda create -n DID python=3.7
conda activate DID
```


Clone the project and go into the project directory
```bash
git clone https://github.com/TATQAQOAO/DL2024_Team12_iIlegal_dumping.git
cd DL2024_Team12_iIlegal_dumping/main
```
Install necessary python packages using
```bash
pip install -r requirements.txt
```
Install torch and matched torchvision from [pytorch.org](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#install-linux)

The code was tested using torch 1.11.0+cu113 and torchvision==0.12.0

Next you need to install Pytorch. You can use:

```bash
pip install torch torchvision
```

```python

python train.py --weights yolov7.pt --cfg cfg/training/yolov7_data.yaml --data data/final/data.yaml --batch-size 32 --epoch 10

python detect.py --weights runs\train\exp11\weights\best.pt --source data\detection

python detect.py --weights yolov7.pt --conf 0.25 --img-size 640 --source inference\images\horses.jpg

python detect.py --weights yolov7.pt --source data\detection
```

