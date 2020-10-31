# UnifiedQA BART

This is a BART version of [UnifiedQA](https://github.com/allenai/unifiedqa),
based on [PyTorch](https://pytorch.org/) and [Huggingface's Transformers](https://github.com/huggingface/transformers).
It contains codes for UnifiedQA training and finetuning on test datasets.

For details of the model and training procedure, please refer to this paper:

Daniel Khashabi, Sewon Min, Tushar Khot, Ashish Sabharwal, Oyvind Tafjord, Peter Clark, Hannaneh Hajishirzi, [UnifiedQA: Crossing Format Boundaries With a Single QA System](https://arxiv.org/abs/2005.00700).
Findings of EMNLP (long). 2020.

```
@inproceedings{ khashabi2020unifiedqa,
    title={ {U}nified{QA}: Crossing Format Boundaries With a Single QA System },
    author={ Khashabi, Daniel and Min, Sewon and Khot, Tushar and Sabharwal, Ashish and Tafjord, Oyvind and Clark, Peter and Hajishirzi, Hannaneh },
    booktitle={ Findings of EMNLP },
    year={2020}
}
```

## Content
1. [Requirement](#requirement)
2. [Quick test](#quick-test)
3. [UnifiedQA training](#unifiedqa-training)
4. [Finetuning](#finetuning)
5. [Evaluation](#evaluation)
6. [Results](#results)


## Requirement

This code is tested on Python 3.6.9.

Install PyTorch and Transformers:
```
pip install torch==1.1.0
pip install git+https://github.com/huggingface/transformers.git@7b75aa9fa55bee577e2c7403301ed31103125a35
```

Download all UnifiedQA datasets and test dataset, which are preprocessed in an input-output format:
```
chmod +x download_data.sh; ./download_data.sh
```

You can also use try an arbitrary data by properly formatting the data, as [examples here](https://github.com/allenai/unifiedqa#feeding-data-into-unifiedqa).

## Quick test

Before reproducing UnifiedQA, you can try a quick test using the released UnifiedQA checkpoint.

First, download the model checkpoint from [here](https://nlp.cs.washington.edu/ambigqa/models/unifiedQA/unifiedQA-bart.zip) (3.6G, containing both uncased and cased versions; we recommend to use an uncased version because it gives more robust performance).

```python
import torch
from transformers import BartTokenizer
from bart import MyBart

base_model = "facebook/bart-large"
unifiedqa_path = "unifiedQA-uncased/best-model.pt" # path to the downloaded checkpoint

tokenizer = BartTokenizer.from_pretrained(base_model)
model = MyBart.from_pretrained(base_model, state_dict=torch.load(unifiedqa_path))
model.eval()

x = model.generate_from_string("Which is best conductor? \\n (A) iron (B) feather", tokenizer=tokenizer)
print (x)

x = model.generate_from_string("What is the sum of 3 and 5? \\n (A) 8 (B) 3 (C) 5 (D) 10", tokenizer=tokenizer)
print (x)

```

## UnifiedQA training

```
python cli.py --do_train --output_dir out/unifiedqa \
        --is_unifiedqa \
        --train_file data/train.tsv \
        --predict_file data/dev.tsv \
        --train_batch_size ${train_bs} \
        --predict_batch_size ${predict_bs} \
        --append_another_bos --do_lowercase \
        --skip_inference --eval_period 10000
```

This script will save the model checkpoints every 10k steps in `out/unifiedqa`.

If you do not specify `--skip_inference`, it will make an inference on the dev data, print its performance (accuracy) as a log, and only save the checkpoint that gives the best accuracy.
You can specify `--skip_inference` in order to skip the inference and save all checkpoints, which will save training time. You will then run the inference script for each checkpoint on the dev data as a separate thread.

You can use `train_batch_size` and `predict_batch_size` depending on the gpu availability. With one 16GB gpu, you can use `train_batch_size=64, predict_batch_size=64`.
We used `train_batch_size=120` for the experiments in the paper.

## Finetuning

Let `data` the name of the dataset to finetune on. (`ls data` for a list of datasets.)

```
python cli.py --do_train --output_dir out/${data}_unifiedqa \
        --checkpoint ${unifiedqa_checkpoint} \
        --train_file data/${data}/train.tsv \
        --predict_file data/${data}/dev.tsv \
        --train_batch_size ${train_bs} \
        --predict_batch_size ${test_bs} \
        --append_another_bos --do_lowercase
```

The script will save the log and the best checkpoint inside `out/${data}_unifiedqa`.

Other useful commands (please refer to `cli.py` for the full list):
- `eval_period`: interval to evaluate on the dev data
- `verbose`: print a progress bar
- `debug`: train and evaluate on a subset of the dev data for debugging purposes

You can use `train_batch_size` and `predict_batch_size` depending on the gpu availability. With one 16GB gpu, you can use `train_batch_size=64, predict_batch_size=64`.
We used `train_batch_size=120` for the experiments in the paper.

Note:
- This script saves the pre-tokenized data in `data/` once question-answer pairs are tokenized for the first time.
- The model gives the best result when prepending extra BOS token (`--append_another_bos`).
- Inference on multi-gpus is not working for now; we will update the code once it is fixed.
- In order to try a baseline that fine-tunes from initial BART checkpoint without UnifiedQA, you can skip specifying `checkpoint`.


## Evaluation

This evaluation script works for any training (UnifiedQA training, finetuning from UnifiedQA, finetuning from BART).

```
python cli.py --do_predict --output_dir out/${data}_unifiedqa \
        --predict_file data/${data}/dev.tsv \
        --predict_batch_size ${test_bs} \
        --append_another_bos --prefix dev_
python cli.py --do_predict --output_dir out/${data}_unifiedqa \
        --predict_file data/${data}/test.tsv \
        --predict_batch_size ${test_bs} \
        --append_another_bos --prefix test_
```

This command will make inference using the checkpoint `out/${data}_unifiedqa/best-model.pt`, and save the prediction file as `out/${data}/{dev|test}_predictions.json`.
Please use `--checkpoint` and `--prefix` for a specific checkpoint path and prediction file name.


## Results

Below is the table that compares the results with BART and UnifiedQA-BART, both after fine-tuning. The numbers are based on the official metric of each dataset (not the Exact Match accuracy, as the code prints out). Please refer to the paper for details.


|   | OBQA | OBQA w/ ir  | ARC-easy  | ARC-easy w/ IR  | ARC_chal  | ARC-chal w/ IR | QASC | QASC w/ ir |
|---|---|---|---|---|---|---|---|---|
|BART     |67.8|66.2|64.1|79.6|36.6|40.4|50.0|75.3|
|UnifiedQA|63.8|70.0|68.0|82.7|52.1|55.0|53.2|78.2|

|   |RACE|ComQA|WG|PIQA|SIQA|ROPES|Nat w/ ir|
|---|---|---|---|---|---|---|---|
|BART     |78.8|62.5|62.4|77.4|74.0|60.5|42.1|
|UnifiedQA|79.4|64.0|63.6|77.9|73.2|60.0|44.5|




