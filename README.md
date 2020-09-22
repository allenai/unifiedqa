# UnifiedQA


You may want to check out: 
 - Our paper: https://arxiv.org/abs/2005.00700
 - Our demo: https://unifiedqa.apps.allenai.org/


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
The `uncased` models uslaly gave us better and more robust results. 

 - UnifiedQA (BART,large,uncased) [gs://unifiedqa/models/bart/unifiedQA-uncased-xbos-120-resumed/](https://console.cloud.google.com/storage/browser/unifiedqa/models/bart/unifiedQA-uncased-xbos-120-resumed/)  
 - UnifiedQA (BART,large,cased) [gs://unifiedqa/models/bart/unifiedQA-cased-xbos-120-resumed/](https://console.cloud.google.com/storage/browser/unifiedqa/models/bart/unifiedQA-cased-xbos-120-resumed/)

## Feeding data into UnifiedQA
Datasets should be converted into a textin/text-out format. 

 - We use `\n` separators between different parts of the input. This ensures having a humanlike encoding while not making it overly-specific to a certain format. 
 - Question always comes first. 

Here are several examples: 

|  **Dataset** | **SQuAD 1.1 (extractive QA)** |
| :---: | :--- |
|  **Encoded Input** | At what speed did the turbine operate? \n (Nikola_Tesla) On his 50th birthday in 1906, Tesla demonstrated his 200 horsepower (150 kilowatts) 16,000 rpm bladeless turbine. ... |
|  **Encoded Output** | 16,000 rpm |
|  **Dataset** | **NarrativeQA (Abstractive QA)** |
|  **Encoded Input** | What does a drink from narcissus's spring cause the drinker to do?  \n  Mercury has awakened Echo, who weeps for Narcissus, and states that a drink from Narcissus's spring causes the drinkers to ''Grow dotingly enamored of themselves.'' ... |
|  **Encoded Output** | fall in love with themselves |
|  **Dataset** | **ARC-challenge (Multiple-choice QA)** |
|  **Encoded Input** | What does photosynthesis produce that helps plants grow? \n (A) water (B) oxygen (C) protein (D) sugar |
|  **Encoded Output** | sugar |
|  **Dataset** | **MCTest (Multiple-choice QA)** |
|  **Encoded Input** | Who was Billy? \n (A) The skinny kid (B) A teacher (C) A little kid (D) The big kid \n Billy was like a king on the school yard. A king without a queen. He was the biggest kid in our grade, so he made all the rules during recess. ... |
|  **Encoded Output** | The big kid |
|  **Dataset** | **BoolQ (Yes-no QA)** |
|  **Encoded Input** | Was America the first country to have a president?  \n (President) The first usage of the word president to denote the highest official in a government was during the Commonwealth of England ... |
|  **Encoded Output** | no |

If you wanna see how this encoding is done on our datasets, check out this [script](encode_datasets.py). 


### The datasets/tasks used in the experiments
While the datasets we used are all public, it could be a bit time-confusing to convert them all into text-to-text format. We're releasing the already-proccessed text-to-text datasets based on the encoding used in this work. Files are included in [this Google Cloud bucket](https://console.cloud.google.com/storage/browser/unifiedqa/data). [Here](encode_datasets.py) is the script we used in order to convert each dataset into text-in-text-out format. 

## Prediction files 
We're making the predictions of the many of our models available. 
[To be updated]


## Using the models in PyTorch/HuggingFace


First download a subset of the model directory, as we need need only few of the files:
```
$ ls -l
total 316048
-rw-r--r--  1 tafjord  staff        387 May 12 18:51 checkpoint
-rw-r--r--  1 tafjord  staff          8 May 12 18:01 model.ckpt-1100500.data-00000-of-00002
-rw-r--r--  1 tafjord  staff  121752064 May 12 18:01 model.ckpt-1100500.data-00001-of-00002
-rw-r--r--  1 tafjord  staff       5677 May 12 18:01 model.ckpt-1100500.index
-rw-r--r--  1 tafjord  staff   26892327 May 12 18:02 model.ckpt-1100500.meta
```
Then edit the checkpoint file so it refers to the right checkpoint in the first line:
```
$ cat checkpoint 
model_checkpoint_path: "model.ckpt-1100500"
all_model_checkpoint_paths: "model.ckpt-1000000"
all_model_checkpoint_paths: "model.ckpt-1020100"
...
```

Now, can run this with Transformers 2.9:

```python
from transformers import T5Config, T5Tokenizer, T5ForConditionalGeneration
from transformers.modeling_t5 import load_tf_weights_in_t5

base_model = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(base_model)
model = T5ForConditionalGeneration(T5Config.from_pretrained(base_model))

load_tf_weights_in_t5(model, None, "/Users/tafjord/models/t5/unifiedqa-small/")
model.eval()

def run_model(input_string, **generator_args):
    input_ids = tokenizer.encode(input_string, return_tensors="pt")
    res = model.generate(input_ids, **generator_args)
    return [tokenizer.decode(x) for x in res]
```

This agrees with the output of T5 code:

```python
run_model("Which is best conductor? \n (A) iron (B) feather")
```
which gives: `['iron']`


```python 
run_model("Scott filled a tray with juice and put it in a freezer. The next day, Scott opened the freezer. How did the juice most likely change? \n (A) It condensed. (B) It evaporated. (C) It became a gas. (D) It became a solid.")
```
which produces: `['it condensed.']`. 


Note that you can also pass in the arguments for text generation to the `run_model(.)` function: 
```python 
run_model("Which is best conductor? \n (A) iron (B) feather (C) wood (D) plastic",
         temperature=0.9, num_return_sequences=4, num_beams=20)
```


## FAQ
**I am not getting the expected results.** An common issue with using UnifiedQA is making sure you use the separator (`\n`) when encoding encoding your inputs. See [the earlier section](#feeding-data-into-unifiedqa) where we delineate how to encode the inputs. 


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

