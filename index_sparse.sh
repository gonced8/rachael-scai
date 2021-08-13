#!/bin/bash

if [[ $# -lt 1 ]]
then
	echo "$0 <collection_path>"
	exit 2
fi

input_directory="$1"
index_directory="$1/index/sparse"

python -m pyserini.index -collection JsonCollection \
                         -generator DefaultLuceneDocumentGenerator \
                         -threads 16 \
                         -input "$input_directory" \
                         -index "$index_directory" \
                         -storePositions -storeDocvectors -storeRaw
