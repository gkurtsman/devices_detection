#! /bin/sh

$OTE/ote_learn_venv/bin/ote train \
$OTE/mmdetection/configs/custom-object-detection/gen3_mobilenetV2_SSD/template_experimental.yaml \
--train-ann-files $OTE/data/vehicle_detection/annotation_train.json \
--train-data-roots $OTE/data/vehicle_detection/train/ \
--val-ann-files $OTE/data/vehicle_detection/annotation_val.json \
--val-data-roots $OTE/data/vehicle_detection/val \
--save-model-to $GITROOT/truck_learning/ote_output_model/ \
params --learning_parameters.batch_size 1 \
--learning_parameters.num_iters 10

# ... external/model-preparation-algorithm/configs/detection/mobilenetv2_ssd_cls_incr/truck_template.yaml
