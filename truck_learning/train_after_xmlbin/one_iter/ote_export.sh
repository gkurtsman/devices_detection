#! /bin/sh

#python -m pdb /home/gena/Develop/comp_vision/dnn/Openvino/training_extensions/truck_detect_venv/bin/ote export /home/gena/Develop/comp_vision/dnn/Openvino/training_extensions/external/mmdetection/configs/custom-object-detection/gen3_mobilenetV2_SSD/template_experimental.yaml --load-weights /home/gena/Develop/comp_vision/dnn/openvino_dev/devices_detection/truck_learning/train_after_xmlbin/one_iter/weights.pth --save-model-to /home/gena/tmp

/home/gena/Develop/comp_vision/dnn/Openvino/training_extensions/truck_detect_venv/bin/ote export \
/home/gena/Develop/comp_vision/dnn/Openvino/training_extensions/external/mmdetection/configs/custom-object-detection/gen3_mobilenetV2_SSD/template_experimental.yaml \
--load-weights /home/gena/Develop/comp_vision/dnn/openvino_dev/devices_detection/truck_learning/train_after_xmlbin/one_iter/weights.pth \
--save-model-to /home/gena/Develop/comp_vision/dnn/openvino_dev/devices_detection/truck_learning/train_after_xmlbin/one_iter/export

