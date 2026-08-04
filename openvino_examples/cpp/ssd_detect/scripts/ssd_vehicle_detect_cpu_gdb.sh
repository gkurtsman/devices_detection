#!/bin/bash

gdb --args $OPENVINO/samples/cpp/build_samples/intel64/Debug/object_detection_sample_ssd -m /opt/intel/openvino_install_git/openvino_env/lib/python3.8/site-packages/openvino/model_zoo/models/intel/vehicle-detection-0202/FP32/vehicle-detection-0202.xml -i /home/gena/Develop/comp_vision/workspace/pc_images/convert_volexcUP_392_36/ir/_Infrared_2_1569186912963.61425781250000_bright.png -d CPU
