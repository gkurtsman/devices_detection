#!/bin/sh

/home/gena/Develop/comp_vision/dnn/Openvino/omz_demos_build/intel64/Release/object_detection_demo \
-at ssd \
-i  /home/gena/Develop/comp_vision/workspace/pc_images/convert_volexcUP_392_36/ir/_Infrared_2_1569186912963.61425781250000_bright.png \
-m /opt/intel/openvino_2021.4.582/deployment_tools/open_model_zoo/tools/downloader/public/ssd300/FP32/ssd300.xml \
-o bucket.bmp \
-d CPU

