#!/bin/bash
if [ -z $1 ]; then
printf "usage: \n %s <directory name> <filename>\n" $0
exit 1
fi

declare FL1=$1$2
echo "dir_base=" $1
echo "filename=" $2
echo $FL1

$OPENVINO/samples/cpp/object_detection_sample_ssd/intel64/Debug/object_detection_sample_ssd \
-m  /home/gena/Develop/comp_vision/dnn/openvino_dev/devices_detection/model/ote_trainingExt_models/vehicle-detection-0204_weights/openvino.xml -i /home/gena/Develop/comp_vision/workspace/pc_images/convert_volexcUP_392_36/ir/_Infrared_2_1569186912963.61425781250000_bright.png -d CPU


#-i '/home/gena/Develop/comp_vision/workspace/pc_images/convert_volexcUP_392_36/ir/_Infrared_2_1569186912963.61425781250000_bright.png' \
#xml at openvino git install:
#$OPENVINO/openvino_env/lib/python3.8/site-packages/openvino/model_zoo/models/intel/vehicle-detection-0202/FP32/vehicle-detection-0202.xml
