#! /bin/sh

$OTE/truck_detect_venv/bin/ote train \
$OTE/external/mmdetection/configs/custom-object-detection/gen3_mobilenetV2_SSD/template_experimental.yaml \
--train-ann-files  $OTE/data/car_tree_bug/annotations/instances_default.json \
--train-data-roots $OTE_XMLBIN/img \
--val-ann-files  $OTE/data/car_tree_bug/annotations/instances_default.json \
--val-data-roots $OTE_XMLBIN/img_val \
--save-model-to $OTE_XMLBIN/../aux \
params --learning_parameters.batch_size 1 \
--learning_parameters.num_iters 1

# ... external/model-preparation-algorithm/configs/detection/mobilenetv2_ssd_cls_incr/truck_template.yaml
