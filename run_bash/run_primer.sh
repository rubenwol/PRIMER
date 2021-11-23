#!/bin/bash

cd ../script

DATA_NAME="multi_news"
# original PRIMER model
MODEL_NAME="PRIMER"
MODEL_PATH = "allenai/led-large-16384"

for RAND_SEED in 1111 1234 5555 6789 7362
do
python primer_main.py  \
                --gpus 1  \
                --mode train \
                --lr 3e-5 \
                --label_smoothing 0.1 \
                --accum_data_per_step 1 \
                --warmup_steps 10000 \
                --total_steps 100000 \
                --batch_size 16 \
                --model_path ${MODEL_NAME}/  \
                --primer_path ${MODEL_PATH} \
                --progress_bar_refresh_rate 50 \
                --rand_seed ${RAND_SEED} \
                --saveTopK 3 \
                --test_imediate \
                --test_batch_size 8 \
                --grad_ckpt \
        > ../train_${DATA_NAME}_${RAND_SEED}.out &
done