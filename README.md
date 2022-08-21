# molske-model

This is a recipe for training a model for [molske](https://github.com/yamnor/molske) to detect molecules drawn with _hexagon-shaped_ **atoms** and _hand-drawn_ **bonds** using the YOLOv5 algorithm on the Google Colab server.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yamnor/molske-model/blob/main/molske_model.ipynb)


```
!git clone https://github.com/yamnor/molske-model
%cd molske-model
```

```
!python molske-model.py 1000
```

```
!git clone https://github.com/ultralytics/yolov5
%cd yolov5
!pip install -r requirements.txt
%cd..
```

```
!pip install wandb
```

```
!python yolov5/train.py --data dataset/data.yaml --cfg yolov5s.yaml --weights '' --img 640  --epochs 100 --batch-size 64 --device 0 --project 'molske' --name 'train' --exist-ok
```

```
!python yolov5/detect.py --source example.jpg --conf 0.5 --weights 'molske/train/weights/best.pt' --project 'molske' --name 'detect' --exist-ok
```