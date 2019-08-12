#! /bin/bash

source "/home/vencintgamer_gmail_com/anaconda3/bin/activate" tensorflow_gpu
python --version

echo "$PWD"
export PYTHONPATH=$PYTHONPATH:"$PWD":"$PWD/slim"

CONFIG_FILE='export/pipeline.config'
CHECKPOINT_PATH='export/model.ckpt'
OUTPUT_DIR='/tmp/tflite'

python object_detection/export_tflite_ssd_graph.py \
	--pipeline_config_path=$CONFIG_FILE \
	--trained_checkpoint_prefix=$CHECKPOINT_PATH \
	--output_directory=$OUTPUT_DIR \
	--add_postprocessing_op=true

cd ../tensorflow

bazel run --config=opt tensorflow/lite/toco:toco -- \
--input_file=$OUTPUT_DIR/tflite_graph.pb \
--output_file=$OUTPUT_DIR/detect.tflite \
--input_shapes=1,300,300,3 \
--input_arrays=normalized_input_image_tensor \
--output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3'  \
--inference_type=FLOAT \
--allow_custom_ops
