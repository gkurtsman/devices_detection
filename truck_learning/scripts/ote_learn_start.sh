#! /bin/bash

source /opt/intel/openvino_install_git/openvino_env/bin/activate
source /opt/intel/openvino_install_git/setupvars.sh
source /home/gena/Develop/comp_vision/dnn/Openvino/training_extensions/ote_learn_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$OTE/ote_sdk:$OTE/external/mmdetection:$OTE/external/mmdetection/submodule:/opt/intel/open_model_zoo/demos/common/python/
export PATH=$PATH:/opt/intel/openvino_install_git/openvino_env/bin/
