#! /bin/sh

/home/gena/Develop/comp_vision/dnn/Openvino/training_extensions/truck_detect_venv/bin/ote export /home/gena/Develop/comp_vision/dnn/Openvino/training_extensions/external/mmdetection/configs/custom-object-detection/gen3_mobilenetV2_SSD/template_experimental.yaml \
--load-weights /home/gena/Develop/comp_vision/dnn/openvino_dev/devices_detection/truck_learning/ote_output_model/ote_train_noWeights_out/weights.pth \
--save-model-to /home/gena/Develop/comp_vision/dnn/openvino_dev/devices_detection/model/ote_trainingExt_models/vehicle-detection-0204_weights

# --load-weights /home/gena/Develop/comp_vision/dnn/openvino_dev/devices_detection/truck_learning/ote_output_model/ote_train_noWeights_out/weights.pth \
