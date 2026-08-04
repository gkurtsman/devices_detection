#! /bin/sh

$OTE/truck_detect_venv/bin/ote train \
$OTE/external/mmdetection/configs/custom-object-detection/gen3_mobilenetV2_SSD/template_experimental.yaml \
--train-ann-files $OTE/data/vehicle_detection/annotation_train.json \
--train-data-roots $OTE/data/vehicle_detection/train/ \
--val-ann-files $OTE/data/vehicle_detection/annotation_val.json \
--val-data-roots $OTE/data/vehicle_detection/val \
--save-model-to $GITROOT/truck_learning/ote_output_model/ \
params --learning_parameters.batch_size 1 \
--learning_parameters.num_iters 10

# \
#--save-model-to /home/gena/Develop/comp_vision/dnn/openvino_dev/devices_detection/truck_learning/ote_output_model/ \
# ... external/model-preparation-algorithm/configs/detection/mobilenetv2_ssd_cls_incr/truck_template.yaml
# /home/gena/Develop/comp_vision/dnn/Openvino/training_extensions/external/mmdetection/configs/custom-object-detection/gen3_mobilenetV2_SSD/template_experimental.yaml \
