#!/usr/bin/env sh
# Compute the mean image from lmdb
# Source: caffe/examples/imagenet/make_imagenet_mean.sh

if [[ -z $1 ]] || [[ -z $2 ]] || [[ -z $3 ]]
then
echo "usage: <SCRIPT_NAME> <full path to lmdb dir> <.binaryproto file name with full path>" \
		"<full path to compute_image_mean utility>"
fi 

LMDB_DIR=$1
MEAN_FILE_NAME=$2
TOOLS=$3

$TOOLS/compute_image_mean $LMDB_DIR \
  $MEAN_FILE_NAME

echo "Done."
