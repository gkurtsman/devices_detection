#!/bin/sh

/opt/intel/open_model_zoo/demos/build/intel64/Debug/object_detection_demo \
-at ssd \
-i /home/gena/Develop/comp_vision/workspace/pc_images/convert_volexcUP_392_36/ir/_Infrared_2_1569186912963.61425781250000_bright.png \
-m  /home/gena/Develop/comp_vision/dnn/openvino_dev/devices_detection/model/ote_trainingExt_models/vehicle-detection-0204_weights/openvino.xml \
-o /home/gena/Develop/comp_vision/dnn/openvino_dev//devices_detection/openvino_examples/cpp/ssd_detect/output_files/out.bmp \
-d CPU

# -i /home/gena/Develop/comp_vision/dnn/openvino_dev/vehicle_images/bus.jpg \
# -m /home/gena/Develop/comp_vision/dnn/openvino_dev/devices_detection/model/ote_trainingExt_models/vehicle-detection-0204_weights/openvino.xml \
