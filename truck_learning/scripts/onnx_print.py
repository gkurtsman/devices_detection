import torch
import onnx
import pickle

onnx_model_path = '/home/gena/Develop/comp_vision/dnn/openvino_dev/devices_detection/truck_learning/ote_output_model/auxiliary_files/model.onnx'
save_path = '/home/gena/Develop/comp_vision/dnn/openvino_dev/devices_detection/truck_learning/ote_output_model/auxiliary_files/model.onnx.saved'
onnx_model = onnx.load(onnx_model_path)
print(onnx_model)
with open(save_path, 'wb') as sp:
    pickle.dump(onnx_model, sp)

# truck_learning/train_after_xmlbin/aux/model.onnx
