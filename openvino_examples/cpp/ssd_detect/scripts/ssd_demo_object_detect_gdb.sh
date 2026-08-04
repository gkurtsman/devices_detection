#!/bin/sh

gdb --args /opt/intel/open_model_zoo/demos/build/intel64/Debug/object_detection_demo -at ssd -i /home/gena/Develop/comp_vision/workspace/pc_images/convert_volexcUP_392_36/ir/_Infrared_2_1569186912963.61425781250000_bright.png -m /opt/intel/open_model_zoo/models/intel/vehicle-detection-0202/FP32/vehicle-detection-0202.xml -o /home/gena/Develop/comp_vision/dnn/openvino_dev//devices_detection/openvino_examples/cpp/ssd_detect/output_files/out.bmp -d CPU


