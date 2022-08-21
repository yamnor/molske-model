# molske-model

This is a recipe for training a model to detect molecules drawn with _hexagon-shaped_ **atoms** and _hand-drawn_ **bonds** using the YOLOv5 algorithm on the Google Colab server.

```
!git clone https://github.com/yamnor/molske-model
!cd molske-model
!python molske-model.py 1000
```

```
!git clone https://github.com/ultralytics/yolov5
!cd yolov5
!pip install -r requirements.txt
!cd..
```

```
!pip install wandb
!wandb online
```

```
!python yolov5/train.py --data dataset/data.yaml --cfg yolov5s.yaml --weights '' --img 640  --epochs 100 --batch-size 64 --device 0 --project 'molske' --name 'train' --exist-ok
```

```
!python yolov5/detect.py --source test.jpg --conf 0.5 --weights 'molcaptor/train/weights/best.pt' --project 'molske' --name 'detect' --exist-ok
```