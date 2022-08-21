# Train moldraw dataset with YOLOv5

!git clone https://github.com/ultralytics/yolov5
%cd yolov5
!pip install -r requirements.txt
%cd ..

!pip install wandb
!wandb online

!python moldraw.py 1000

!python yolov5/train.py --data dataset/data.yaml --cfg yolov5s.yaml --weights '' --img 640  --epochs 100 --batch-size 64 --device 0 --project 'molcaptor' --name 'train' --exist-ok

!python yolov5/detect.py --source test.jpg --conf 0.5 --weights 'molcaptor/train/weights/best.pt' --project 'molcaptor' --name 'detect' --exist-ok
