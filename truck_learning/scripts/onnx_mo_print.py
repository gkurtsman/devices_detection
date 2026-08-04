import torch
import onnx
import pickle

onnx_model_path = '/home/gena/Develop/comp_vision/dnn/openvino_dev/devices_detection/truck_learning/train_after_xmlbin/aux/model_beforeMO.onnx'
onnx_model = onnx.load(onnx_model_path)
