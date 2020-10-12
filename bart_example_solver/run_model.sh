#!/usr/bin/env bash

set -e

python solver.py --input-file physicaliqa.jsonl \
                 --output-file predictions.lst \
                 --model bart-models/physical_iqa-unifiedQA-uncased-xbos-120/best-model-028000.pt
