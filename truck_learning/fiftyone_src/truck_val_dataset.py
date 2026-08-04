import fiftyone as fo

# Create a dataset from a directory of images
dataset = fo.Dataset.from_images_dir("/home/gena/Develop/comp_vision/dnn/devices_detection/truck_learning/dataset_files")
session = fo.launch_app(dataset)
input1 = input()

