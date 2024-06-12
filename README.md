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
conda create -n DID python=3.8.19
conda activate DID
```


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

### Yolov7 training
Download yolov7 model [here](https://github.com/TATQAQOAO/DL2024_Team12_iIlegal_dumping/tree/main/yolov7-main)

```python
python train.py --weights yolov7.pt --cfg cfg/training/yolov7_data.yaml --data data/final/data.yaml --batch-size 32 --epoch 10
```

python detect.py --weights runs\train\exp11\weights\best.pt --source data\detection

python detect.py --weights yolov7.pt --conf 0.25 --img-size 640 --source inference\images\horses.jpg

python detect.py --weights yolov7.pt --source data\detection
```

