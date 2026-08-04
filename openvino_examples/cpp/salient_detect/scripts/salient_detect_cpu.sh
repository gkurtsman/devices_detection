#!/bin/bash
if [ -z $1 ]; then
printf "usage: \n %s <directory name> <filename>\n" $0
exit 1
fi

declare FL1=$1$2
echo "dir_base=" $1
echo "filename=" $2
echo $FL1

/home/gena/Develop/comp_vision/dnn/openvino_dev/devices_detection/openvino_examples/cpp/salient_detect/src/build/salient \
-m /opt/intel/openvino_2021/deployment_tools/open_model_zoo/tools/downloader/public/f3net/FP32/f3net.xml \
-i $FL1 \
-d CPU
#-m /opt/intel/openvino_2021/deployment_tools/open_model_zoo/tools/downloader/public/f3net/FP32/f3net.xml \
#-i /home/gena/Develop/comp_vision/workspace/pc_images/convert_volexcUP_392_36/ir/_Infrared_2_1569186912963.61425781250000_bright.png \
# -i '/home/gena/Develop/comp_vision/workspace/pc_images/convert_volexcUP_382_35/ir/_Infrared 2_1569147841103.50244140625000.png' \
# -i /home/gena/Develop/comp_vision/workspace/pc_images/convert_volexcUP_392_36/color/_Color_1569186900979.54492187500000.png \
#-i '/home/gena/Develop/comp_vision/workspace/pc_images/convert_volexcUP_700_41/ir/_Infrared 2_1569215417645.03247070312500.png' \
