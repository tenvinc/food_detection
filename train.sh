#! /bin/bash

source "/home/vencintgamer_gmail_com/anaconda3/bin/activate" tensorflow_gpu
python --version

echo "$PWD"
export PYTHONPATH=$PYTHONPATH:"$PWD":"$PWD/slim"

PIPELINE_CONFIG_PATH="food_detection.config"
MODEL_DIR="training/"
NUM_TRAIN_STEPS=50000
python object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --alsologtostderr 
#    --checkpoint_dir=training \
#    --eval_training_data=True
