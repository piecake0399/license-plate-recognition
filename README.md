# license plate recognition
 License Plate Recognition system using YOLOv5 and EasyOCR
# Usage
Installing Anaconda virtual enviroment: https://www.anaconda.com/download

Create enviroment for YOLO model

```
conda create --name yolo-env1 python=3.10 -y 
conda activate yolo-env1
```

Installing packages

```
pip install requirements.txt
```

Run:

```
python main.py --source <path to your source>
```
path to your source can be path of an image, folder of images, video or webcam (usb0)
