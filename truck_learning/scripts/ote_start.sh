#! /bin/bash

source $OPENVINO/openvino_env/bin/activate
source $OPENVINO/setupvars.sh
source $OTE/truck_detect_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$OTE/ote_sdk:$OTE/external/mmdetection/
export PYTHONPATH=$PYTHONPATH:$OTE/external/mmdetection/submodule/
export PYTHONPATH=$PYTHONPATH:$ZOO/demos/common/python/
export PYTHONPATH=$PYTHONPATH:/usr/local/cuda-11.1/targets/x86_64-linux/lib/
export PATH=$PATH:$OPENVINO/openvino_env/bin/
export PYTHONPATH=$PYTHONPATH:$OPENVINO/openvino_env/lib/python3.8/site-packages

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.1
