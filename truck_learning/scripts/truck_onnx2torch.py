import torch
from onnx2torch import convert

# Path to ONNX model
onnx_model_path = '/home/gena/Develop/comp_vision/dnn/openvino_dev/devices_detection/truck_learning/ote_output_model/auxiliary_files/model.old.onnx'
# You can pass the path to the onnx model to convert it or...
torch_model_1 = convert(onnx_model_path)
print (torch_model_1)
torch.save(torch_model_1, "/home/gena/Develop/comp_vision/dnn/openvino_dev/devices_detection/truck_learning/ote_output_model/auxiliary_files/torch2onnx.pth")

