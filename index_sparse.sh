#!/bin/bash

python -m pyserini.index -collection JsonCollection \
                         -generator DefaultLuceneDocumentGenerator \
                         -threads 16 \
                         -input data/ubuntu/documents \
                         -index data/ubuntu/index/sparse \
                         -storePositions -storeDocvectors -storeRaw
