#! /bin/sh

#/opt/intel/openvino_install_git/openvino_env/bin/mo \
python /opt/intel/openvino_install_git/./openvino_env/lib/python3.8/site-packages/openvino/tools/mo/main.py \
--input_model=/home/gena/Develop/comp_vision/dnn/openvino_dev/devices_detection/truck_learning/ote_output_model/auxiliary_files/model.onnx \
'--mean_values=[0, 0, 0] ' '--scale_values=[255, 255, 255] ' \
--output_dir=/home/gena/Develop/comp_vision/dnn/openvino_dev/devices_detection/truck_learning/ote_output_model/onnx2binxml \
--output=feature_vector,saliency_map,labels,boxes --data_type=FP32 '--input_shape=[1, 3, 864, 864]'
