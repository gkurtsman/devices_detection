    # Object Detection

    ## Import everything

import os
import sys

import cv2
import numpy as np
from ote_sdk.configuration.helper import create as create_parameters_from_parameters_schema
from ote_sdk.entities.inference_parameters import InferenceParameters
from ote_sdk.entities.label_schema import LabelSchemaEntity
from ote_sdk.entities.model import ModelEntity
from ote_sdk.entities.resultset import ResultSetEntity
from ote_sdk.entities.subset import Subset
from ote_sdk.entities.task_environment import TaskEnvironment
from ote_sdk.usecases.tasks.interfaces.export_interface import ExportType

from ote_cli.datasets import get_dataset_class
from ote_cli.registry import Registry
from ote_cli.utils.importing import get_impl_class
## Register templates
templates_dir = '/home/gena/Develop/comp_vision/dnn/Openvino/training_extensions/external'
registry = Registry(templates_dir)
print("registry:\n", registry, "registry completed\n")
## Load model template and its hyper parameters
model_template = registry.get('Custom_Object_Detection_Gen3_ATSS')
hyper_parameters = model_template.hyper_parameters.data

    ## Get dataset instantiated

Dataset = get_dataset_class(model_template.task_type)

dataset = Dataset(
    train_subset={'ann_file': '/home/gena/Develop/comp_vision/dnn/Openvino/training_extensions/data/airport/annotation_faces_train.json',
      'data_root': '/home/gena/Develop/comp_vision/dnn/Openvino/training_extensions/data/airport'},
    val_subset={'ann_file': '/home/gena/Develop/comp_vision/dnn/Openvino/training_extensions/data/airport/annotation_faces_train.json',
        'data_root': '/home/gena/Develop/comp_vision/dnn/Openvino/training_extensions/data/airport/'}
)
labels_schema = LabelSchemaEntity.from_labels(dataset.get_labels())
## Have a look at existing parameters
hyper_parameters = create_parameters_from_parameters_schema(hyper_parameters)
 
for p in hyper_parameters.learning_parameters.parameters:
    print(f'{p}: {getattr(hyper_parameters.learning_parameters, p)}')

    ## Tweak parameters

hyper_parameters.learning_parameters.batch_size = 8
hyper_parameters.learning_parameters.num_iters = 5

for p in hyper_parameters.learning_parameters.parameters:
    print(f'{p}: {getattr(hyper_parameters.learning_parameters, p)}')

    ## Create Task 

Task = get_impl_class(model_template.entrypoints.base)

environment = TaskEnvironment(
    model=None,
    hyper_parameters=hyper_parameters,
    label_schema=labels_schema,
    model_template=model_template)

task = Task(task_environment=environment)

    ## Run training

output_model = ModelEntity(
    dataset,
    environment.get_model_configuration(),
)

task.train(dataset, output_model)

    ## Evaluate quality metric

validation_dataset = dataset.get_subset(Subset.VALIDATION)
predicted_validation_dataset = task.infer(
    validation_dataset.with_empty_annotations(),
    InferenceParameters(is_evaluation=True))

resultset = ResultSetEntity(
    model=output_model,
    ground_truth_dataset=validation_dataset,
    prediction_dataset=predicted_validation_dataset,
)
task.evaluate(resultset)
print(resultset.performance)
## Export model to OpenVINO format
exported_model = ModelEntity(
    dataset,
    environment.get_model_configuration()
)
task.export(ExportType.OPENVINO, exported_model)

    ## Evaluate the exported model

environment.model = exported_model
ov_task = get_impl_class(model_template.entrypoints.openvino)(environment)
predicted_validation_dataset = ov_task.infer(
    validation_dataset.with_empty_annotations(),
    InferenceParameters(is_evaluation=True))
resultset = ResultSetEntity(
    model=output_model,
    ground_truth_dataset=validation_dataset,
    prediction_dataset=predicted_validation_dataset,
)
ov_task.evaluate(resultset)
print(resultset.performance)

    ## Draw bounding boxes

import IPython
import PIL

for predictions, item in zip(predicted_validation_dataset, validation_dataset.with_empty_annotations()):
    image = item.numpy.astype(np.uint8)
    for box in predictions.annotation_scene.shapes:
        x1 = int(box.x1 * image.shape[1])
        x2 = int(box.x2 * image.shape[1])
        y1 = int(box.y1 * image.shape[0])
        y2 = int(box.y2 * image.shape[0])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)  
    IPython.display.display(PIL.Image.fromarray(image))
