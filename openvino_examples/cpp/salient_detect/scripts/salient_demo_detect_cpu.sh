/home/gena/Develop/comp_vision/dnn/Openvino/omz_demos_build/intel64/Release/object_detection_demo \
-at ssd \
-m /opt/intel/openvino_2021/deployment_tools/open_model_zoo/tools/downloader/public/f3net/FP32/f3net.xml \
-i '/home/gena/Develop/comp_vision/workspace/pc_images/convert_volexcUP_392_36/color/_Color_1569186900979.54492187500000.png' \
-o bucket.bmp \
-auto_resize \
-d CPU
# -i '/home/gena/Develop/comp_vision/workspace/pc_images/convert_volexcUP_382_35/ir/_Infrared 2_1569147841103.50244140625000.png' \
