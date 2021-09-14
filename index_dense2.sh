#!/bin/bash

cd "$(dirname "$0")"
source env/bin/activate

cd "$(dirname "$0")"

encoder="sentence-transformers/msmarco-distilbert-base-v3"

((CUDA_VISIBLE_DEVICES=2 python -m pyserini.dindex --corpus data/qrecc/collection/ --encoder $encoder --index data/qrecc/dense_index-0 --batch 64 --device cuda:0 --shard-id 0 --shard-num 10 > index-0.txt 2>&1);
(CUDA_VISIBLE_DEVICES=2 python -m pyserini.dindex --corpus data/qrecc/collection/ --encoder $encoder --index data/qrecc/dense_index-1 --batch 64 --device cuda:0 --shard-id 1 --shard-num 10 > index-1.txt 2>&1);
(CUDA_VISIBLE_DEVICES=2 python -m pyserini.dindex --corpus data/qrecc/collection/ --encoder $encoder --index data/qrecc/dense_index-2 --batch 64 --device cuda:0 --shard-id 2 --shard-num 10 > index-2.txt 2>&1);
(CUDA_VISIBLE_DEVICES=2 python -m pyserini.dindex --corpus data/qrecc/collection/ --encoder $encoder --index data/qrecc/dense_index-3 --batch 64 --device cuda:0 --shard-id 3 --shard-num 10 > index-3.txt 2>&1);
(CUDA_VISIBLE_DEVICES=2 python -m pyserini.dindex --corpus data/qrecc/collection/ --encoder $encoder --index data/qrecc/dense_index-4 --batch 64 --device cuda:0 --shard-id 4 --shard-num 10 > index-4.txt 2>&1)) &

((CUDA_VISIBLE_DEVICES=6 python -m pyserini.dindex --corpus data/qrecc/collection/ --encoder $encoder --index data/qrecc/dense_index-5 --batch 64 --device cuda:0 --shard-id 5 --shard-num 10 > index-5.txt 2>&1);
(CUDA_VISIBLE_DEVICES=6 python -m pyserini.dindex --corpus data/qrecc/collection/ --encoder $encoder --index data/qrecc/dense_index-6 --batch 64 --device cuda:0 --shard-id 6 --shard-num 10 > index-6.txt 2>&1);
(CUDA_VISIBLE_DEVICES=6 python -m pyserini.dindex --corpus data/qrecc/collection/ --encoder $encoder --index data/qrecc/dense_index-7 --batch 64 --device cuda:0 --shard-id 7 --shard-num 10 > index-7.txt 2>&1);
(CUDA_VISIBLE_DEVICES=6 python -m pyserini.dindex --corpus data/qrecc/collection/ --encoder $encoder --index data/qrecc/dense_index-8 --batch 64 --device cuda:0 --shard-id 8 --shard-num 10 > index-8.txt 2>&1);
(CUDA_VISIBLE_DEVICES=6 python -m pyserini.dindex --corpus data/qrecc/collection/ --encoder $encoder --index data/qrecc/dense_index-9 --batch 64 --device cuda:0 --shard-id 9 --shard-num 10 > index-9.txt 2>&1)) &
