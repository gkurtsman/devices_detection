#! /bin/sh

export EXE=/home/gena/Develop/comp_vision/dnn/Openvino/training_extensions/external/mmdetection/submodule

python $EXE/demo/image_demo.py $EXE/tests/data/VOCdevkit/VOC2012/JPEGImages/000001.jpg $EXE/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py $EXE/tests/data/VOCdevkit/VOC2012/Annotations/000001.xml --device cpu
