#bin/sh

/home/krtsmn/Develop/comp_vision/dnn/caffe_ssd/build/examples/ssd/ssd_detect.bin \
/home/krtsmn/Develop/comp_vision/dnn/MobileNet-SSD_models/models/VGGNet/VOC0712/SSD_300x300/deploy.prototxt \
/home/krtsmn/Develop/comp_vision/dnn/MobileNet-SSD_models/models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel \
/home/krtsmn/Develop/comp_vision/dnn/devices_detection/caffe_ssd_object_detect/ssd_list_file.txt \
/home/krtsmn/Develop/comp_vision/dnn/devices_detection/caffe_ssd_object_detect/labels_ssd_caffe.txt
