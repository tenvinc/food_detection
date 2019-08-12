#! /bin/bash

source "/home/vencintgamer_gmail_com/anaconda3/bin/activate" tensorflow_gpu
python --version

echo "$PWD"
export PYTHONPATH=$PYTHONPATH:"$PWD":"$PWD/slim"

INPUT_TYPE=image_tensor
PIPELINE_CONFIG_PATH='food_detection.config'
TRAINED_CKPT_PREFIX='training/model.ckpt-15887'
EXPORT_DIR=export
python object_detection/export_inference_graph.py \
    --input_type=${INPUT_TYPE} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
    --output_directory=${EXPORT_DIR}
