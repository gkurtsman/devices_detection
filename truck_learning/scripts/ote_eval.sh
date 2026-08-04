#! /bin/sh

ote eval \
/home/gena/Develop/comp_vision/dnn/Openvino/training_extensions/external/mmdetection/configs/custom-object-detection/gen3_mobilenetV2_SSD/template_experimental.yaml \
--test-ann-files /home/gena/Develop/comp_vision/dnn/Openvino/training_extensions/data/vehicle_detection/annotation_val.json \
--test-data-roots /home/gena/Develop/comp_vision/dnn/Openvino/training_extensions/data/vehicle_detection/val \
--load-weights /home/gena/Develop/comp_vision/dnn/openvino_dev/devices_detection/truck_learning/ote_optimize_model/weights.pth
