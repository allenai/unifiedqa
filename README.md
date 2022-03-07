# UnifiedQA


You may want to check out:
 - Paper: https://arxiv.org/abs/2005.00700
 - Demo: https://unifiedqa.apps.allenai.org/

Update (Feb '22): UnifiedQA-v2 
 - Paper: https://arxiv.org/abs/2202.12359


## Using the models in PyTorch/HuggingFace

You can very easily load the models with [Transformers](https://github.com/huggingface/transformers/) >=3.1, instead of downloading them manually. 
The models are listed on [this page](https://huggingface.co/allenai). Here is a list of model these model names hosted on HuggingFace model hub: 


| Model Name                | Huggingface ID (s)                      |
|---------------------------|-----------------------------------------|
| UnifiedQA (T5) - small    | `allenai/unifiedqa-t5-small`            |
| UnifiedQA (T5) - base     | `allenai/unifiedqa-t5-base`             |
| UnifiedQA (T5) - large    | `allenai/unifiedqa-t5-large`            |
| UnifiedQA (T5) - 3B       | `allenai/unifiedqa-t5-3b`               |
| UnifiedQA (T5) - 11B      | `allenai/unifiedqa-t5-11b`              |
| UnifiedQA-v2 (T5) - small | `allenai/unifiedqa-v2-t5-small-[ckpt]`  |
| UnifiedQA-v2 (T5) - base  | `allenai/unifiedqa-v2-t5-base-[ckpt]`   |
| UnifiedQA-v2 (T5) - large | `allenai/unifiedqa-v2-t5-large-[ckpt]`  |
| UnifiedQA-v2 (T5) - 3B    | `allenai/unifiedqa-v2-t5-3b-[ckpt]`     |
| UnifiedQA-v2 (T5) - 11B   | `allenai/unifiedqa-v2-t5-11b-[ckpt]`    |


Where `[ckpt]` can be either `1251000` or `1363200`. 

Here is an examples: 

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

model_name = "allenai/unifiedqa-t5-small" # you can specify the model size here
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def run_model(input_string, **generator_args):
    input_ids = tokenizer.encode(input_string, return_tensors="pt")
    res = model.generate(input_ids, **generator_args)
    return tokenizer.batch_decode(res, skip_special_tokens=True)
```

For instance, here is how you can use it to answer a multiple-choice question: 

```python
run_model("which is best conductor? \\n (a) iron (b) feather")
```
which gives: `['iron']`


```python
run_model("scott filled a tray with juice and put it in a freezer. the next day, scott opened the freezer. how did the juice most likely change? \\n (a) it condensed. (b) it evaporated. (c) it became a gas. (d) it became a solid.")
```
which produces: `['it condensed.']`.


Note that you can also pass in the arguments for text generation to the `run_model(.)` function:
```python
run_model("which is best conductor? \\n (a) iron (b) feather (c) wood (d) plastic",
         temperature=0.9, num_return_sequences=4, num_beams=20)
```



## Feeding data into UnifiedQA
Datasets should be converted into a textin/text-out format.

 - Question always comes first.
 - We use `\n` separators between different parts of the input. This ensures having a humanlike encoding while not making it overly-specific to a certain format.  Note that this separator isn't the newline character (which it looks suspiciously like), but rather backslash-n.
 - Make sure the whole input is correctly [pre-processed](https://github.com/allenai/unifiedqa/blob/7bf0653c6fb68a51019924fd4c51615155acbebe/tasks.py#L54-L58) (e.g., lower-cased)

Here are several examples:

|  **Dataset** | **SQuAD 1.1 (extractive QA)** |
| :---: | :--- |
|  **Encoded Input** | `At what speed did the turbine operate? \n (Nikola_Tesla) On his 50th birthday in 1906, Tesla demonstrated his 200 horsepower (150 kilowatts) 16,000 rpm bladeless turbine. ...` |
|  **Encoded Output** | `16,000 rpm` |
|  **Dataset** | **NarrativeQA (Abstractive QA)** |
|  **Encoded Input** | `What does a drink from narcissus's spring cause the drinker to do?  \n  Mercury has awakened Echo, who weeps for Narcissus, and states that a drink from Narcissus's spring causes the drinkers to ''Grow dotingly enamored of themselves.'' ...` |
|  **Encoded Output** | `fall in love with themselves` |
|  **Dataset** | **ARC-challenge (Multiple-choice QA)** |
|  **Encoded Input** | `What does photosynthesis produce that helps plants grow? \n (A) water (B) oxygen (C) protein (D) sugar` |
|  **Encoded Output** | `sugar` |
|  **Dataset** | **MCTest (Multiple-choice QA)** |
|  **Encoded Input** | `Who was Billy? \n (A) The skinny kid (B) A teacher (C) A little kid (D) The big kid \n Billy was like a king on the school yard. A king without a queen. He was the biggest kid in our grade, so he made all the rules during recess. ...` |
|  **Encoded Output** | `The big kid` |
|  **Dataset** | **BoolQ (Yes-no QA)** |
|  **Encoded Input** | `Was America the first country to have a president?  \n (President) The first usage of the word president to denote the highest official in a government was during the Commonwealth of England ...` |
|  **Encoded Output** | `no` |

If you wanna see how this encoding is done on our datasets, check out this [script](encode_datasets.py).


### The datasets/tasks used in the experiments
While the datasets we used are all public, it could be a bit time-confusing to convert them all into text-to-text format. We're releasing the already-proccessed text-to-text datasets based on the encoding used in this work. Files are included in [this Google Cloud bucket](https://console.cloud.google.com/storage/browser/unifiedqa/data). [Here](encode_datasets.py) is the script we used in order to convert each dataset into text-in-text-out format.

## Prediction files
Reach out to DanielK if you want them! :)


## Released Model Checkpoints

If you intend to create a QA system, you can use our QA-specialized models for your purpose:


### T5 models
 - UnifiedQA (T5, small) [gs://unifiedqa/models/small](https://console.cloud.google.com/storage/browser/unifiedqa/models/small)
 - UnifiedQA (T5, base) [gs://unifiedqa/models/base](https://console.cloud.google.com/storage/browser/unifiedqa/models/base)
 - UnifiedQA (T5, large) [gs://unifiedqa/models/large](https://console.cloud.google.com/storage/browser/unifiedqa/models/large)
 - UnifiedQA (T5, 3B) [gs://unifiedqa/models/3B](https://console.cloud.google.com/storage/browser/unifiedqa/models/3B)
 - UnifiedQA (T5, 11B) [gs://unifiedqa/models/11B](https://console.cloud.google.com/storage/browser/unifiedqa/models/11B)

Note: In the experiments reported in our paper we always used the checkpoint closest to 100k steps (it usually corresponds to checkpoint 1100500)

You can use these in two ways:
- If you don't have any training data, you can use them for [the evaluation](https://github.com/google-research/text-to-text-transfer-transformer#eval).
- If you training data, you can use them as your initial models and [fine-tune on them](https://github.com/google-research/text-to-text-transfer-transformer#fine-tuning).

For more details see [the T5 repository](https://github.com/google-research/text-to-text-transfer-transformer).

### BART models
The BART models are downloaded from [this link](https://nlp.cs.washington.edu/ambigqa/models/unifiedQA/unifiedQA-bart.zip) (3.6G).
For detailed instructions on running the code (training/finetuning/testing), please refer to [here](https://github.com/allenai/unifiedqa/tree/master/bart).
The `uncased` models usually gave us better and more robust results.

### v2 T5 models 

 - UnifiedQA v2 (T5, small) [gs://unifiedqa/models_v2/small](https://console.cloud.google.com/storage/browser/unifiedqa/models_v2/small)
 - UnifiedQA v2 (T5, base) [gs://unifiedqa/models_v2/base](https://console.cloud.google.com/storage/browser/unifiedqa/models_v2/base)
 - UnifiedQA v2 (T5, large) [gs://unifiedqa/models_v2/large](https://console.cloud.google.com/storage/browser/unifiedqa/models_v2/large)
 - UnifiedQA v2 (T5, 3B) [gs://unifiedqa/models_v2/3B](https://console.cloud.google.com/storage/browser/unifiedqa/models_v2/3B)
 - UnifiedQA v2 (T5, 11B) [gs://unifiedqa/models_v2/11B](https://console.cloud.google.com/storage/browser/unifiedqa/models_v2/11B)

Note: In the experiments reported in our paper we always used the checkpoint closest to 250k steps. 


## FAQ
**I am not getting the expected results.** An common issue with using UnifiedQA is making sure you use the separator (`\n`) when encoding encoding your inputs. See [the earlier section](#feeding-data-into-unifiedqa) where we delineate how to encode the inputs.

**Help! I am getting the following error!** See [this discussion](https://github.com/google-research/text-to-text-transfer-transformer/issues/180) if you're getting the following error:
```bash
ValueError: Configurable 'make_layer_stack' doesn't have a parameter named 'use_universal_transformer'.
  In file "gs://danielk-files/t5-models/union_mixture/11B/operative_config.gin", line 83
```


## How to cite

If you extend or use this work, please cite the paper:
```bibtex
@article{2020unifiedqa,
    title={UnifiedQA: Crossing Format Boundaries With a Single QA System},
    author={D. Khashabi and S. Min and T. Khot and A. Sabhwaral and O. Tafjord and P. Clark and H. Hajishirzi},
    journal={EMNLP - findings},
    year={2020}
}
```

