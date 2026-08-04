import fiftyone as fo

# Load COCO formatted dataset
coco_dataset = fo.Dataset.from_dir(
    dataset_type=fo.types.COCODetectionDataset,
    data_path="/home/gena/Develop/comp_vision/dnn/devices_detection/truck_learning/dataset_files",
    labels_path="/home/gena/Develop/comp_vision/dnn/devices_detection/truck_learning/dataset_files/instances_default.json",
    include_id=True,
    label_field="",
)

session = fo.launch_app(coco_dataset)
input1 = input()

