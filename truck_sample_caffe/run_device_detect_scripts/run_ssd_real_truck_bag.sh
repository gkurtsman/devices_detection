#bin/sh
if [ -z $1 ]; then
printf "usage: \n %s <directory name> <filename base> <file number (<PAR2PAR3.bag>) sec\n" $0
exit 1
fi

declare FL1=$1$2$3.bag
echo file FL1 = "$FL1"
declare SEC=$4

/home/krtsmn/Develop/comp_vision/dnn/devices_detection/truck_sample_caffe/build/truck_det \
/home/krtsmn/Develop/comp_vision/dnn/MobileNet-SSD_models/models/VGGNet/VOC0712/SSD_300x300/deploy.prototxt \
/home/krtsmn/Develop/comp_vision/dnn/MobileNet-SSD_models/models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel \
/home/krtsmn/Develop/comp_vision/workspace/pc_images/volexcUP_392_36.binaryproto \
/home/krtsmn/Develop/comp_vision/dnn/devices_detection/caffe_ssd_object_detect/labels_ssd_caffe.txt \
$FL1 $SEC
 
