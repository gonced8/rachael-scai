#!/bin/bash

HOME=/tmp/gecr

cd "$(dirname "$0")"
source env/bin/activate

cd "$(dirname "$0")"

CUDA_DEVICE_ORDER="PCI_BUS_ID"

CUDA_VISIBLE_DEVICES=0 python -m pyserini.dindex --corpus data/qrecc/collection/ --encoder castorini/tct_colbert-msmardex --index data/qrecc/dense_index-0 --batch 64 --device cuda:0 --shard-id 0 --shard-num 6 > index-0.txt 2>&1 &
#CUDA_VISIBLE_DEVICES=1 python -m pyserini.dindex --corpus data/qrecc/collection/ --encoder castorini/tct_colbert-msmardex --index data/qrecc/dense_index-1 --batch 64 --device cuda:0 --shard-id 1 --shard-num 6 > index-1.txt 2>&1 &
#CUDA_VISIBLE_DEVICES=2 python -m pyserini.dindex --corpus data/qrecc/collection/ --encoder castorini/tct_colbert-msmardex --index data/qrecc/dense_index-2 --batch 160 --device cuda:0 --shard-id 2 --shard-num 6 > index-2.txt 2>&1 &
#CUDA_VISIBLE_DEVICES=3 python -m pyserini.dindex --corpus data/qrecc/collection/ --encoder castorini/tct_colbert-msmardex --index data/qrecc/dense_index-3 --batch 64 --device cuda:0 --shard-id 3 --shard-num 6 > index-3.txt 2>&1 &
#CUDA_VISIBLE_DEVICES=5 python -m pyserini.dindex --corpus data/qrecc/collection/ --encoder castorini/tct_colbert-msmardex --index data/qrecc/dense_index-4 --batch 160 --device cuda:0 --shard-id 4 --shard-num 6 > index-4.txt 2>&1 &
#CUDA_VISIBLE_DEVICES=7 python -m pyserini.dindex --corpus data/qrecc/collection/ --encoder castorini/tct_colbert-msmardex --index data/qrecc/dense_index-5 --batch 64 --device cuda:0 --shard-id 5 --shard-num 6 > index-5.txt 2>&1 &
