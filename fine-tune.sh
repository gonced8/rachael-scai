#!/bin/sh

if [ -d "env/bin" ] ; then
	alias python="env/bin/python"
fi

python seq2seq/run_summarization.py \
    --model_name_or_path pegasus-large \
    --do_train \
    --do_eval \
    --train_file data/train.csv \
    --validation_file data/validate.csv \
    --output_dir output \
    --overwrite_output_dir \
    --per_device_train_batch_size=256 \
    --per_device_eval_batch_size=256 \
	--max_source_length 512 --max_target_length 64 \
	--learning_rate 1e-4 \
	--freeze_embeds --label_smoothing 0.1 --adafactor \
    --predict_with_generate \
	"$@"
