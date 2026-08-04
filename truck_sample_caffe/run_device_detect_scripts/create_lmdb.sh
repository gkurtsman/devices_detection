#!/usr/bin/sh
# Create the excavator lmdb inputs
# Source: caffe/examples/imagenet/create_imagenet.sh

if [[ -z $1 ]] || [[ -z $2 ]] || [[ -z $3 ]] || [[ -z $4 ]]
then
echo "usage: <SCRIPT_NAME> <full path to images dir> " \
			"<full path to convert_imageset utility> " \
			"<full path to lmdb files dir> <lmdb file name>"
exit 1
fi

set -e

IMAGE_DIR=$1
DATA=$1
TOOLS=$2

TRAIN_DATA_ROOT=$1

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=false
if $RESIZE; then
  RESIZE_HEIGHT=256
  RESIZE_WIDTH=256
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

echo "Creating lmdb file"

rm -rf $3/$4

$TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $TRAIN_DATA_ROOT \
    $DATA/image_list.txt \
    $3/$4

echo "Done."

