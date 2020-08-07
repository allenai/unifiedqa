# UnifiedQA


You may want to check out: 
 - Our paper: https://arxiv.org/abs/2005.00700
 - Our demo: https://unifiedqa.apps.allenai.org/


## Released Model Checkpoints

If you intend to create a QA system, you can use our QA-specialized models for your purpose: 


### T5 models 
 - UnifiedQA (T5,small) [gs://t5-data/unifiedqa/models/small](https://console.cloud.google.com/storage/browser/unifiedqa/models/small)  
 - UnifiedQA (T5,base) [gs://t5-data/unifiedqa/models/base](https://console.cloud.google.com/storage/browser/unifiedqa/models/base)
 - UnifiedQA (T5,large) [gs://t5-data/unifiedqa/models/large](https://console.cloud.google.com/storage/browser/unifiedqa/models/large)
 - UnifiedQA (T5,3B) [gs://t5-data/unifiedqa/models/3B](https://console.cloud.google.com/storage/browser/unifiedqa/models/3B)
 - UnifiedQA (T5,11B) [gs://t5-data/unifiedqa/models/11B](https://console.cloud.google.com/storage/browser/unifiedqa/models/11B)

Note: In the experiments reported in our paper we always used the checkpoint closest to 100k steps (it usually corresponds to checkpoint 1100500) 

You can use these in two ways: 
- If you don't have any training data, you can use them for [the evaluation](https://github.com/google-research/text-to-text-transfer-transformer#eval). 
- If you training data, you can use them as your initial models and [fine-tune on them](https://github.com/google-research/text-to-text-transfer-transformer#fine-tuning).

For more details see [the T5 repository](https://github.com/google-research/text-to-text-transfer-transformer). 

### BART models 
The `uncased` models uslaly gave us better and more robust results. 

 - UnifiedQA (BART,large,uncased) [gs://t5-data/unifiedqa/models/bart/unifiedQA-uncased-xbos-120-resumed/](https://console.cloud.google.com/storage/browser/unifiedqa/models/bart/unifiedQA-uncased-xbos-120-resumed/)  
 - UnifiedQA (BART,large,cased) [gs://t5-data/unifiedqa/models/bart/unifiedQA-cased-xbos-120-resumed/](https://console.cloud.google.com/storage/browser/unifiedqa/models/bart/unifiedQA-cased-xbos-120-resumed/)


## The datasets/tasks used in the experiments
While the datasets we used are all public, it could be a bit time-confusing to convert them all into text-to-text format. We're releasing the already-proccessed text-to-text datasets based on the encoding used in this work. Files are included in [this Google Cloud bucket](https://console.cloud.google.com/storage/browser/unifiedqa/data).  


## Prediction files 
We're making the predictions of the many of our models available. 
[To be updated]

## How to cite

If you extend or use this work, please cite the paper: 
```bibtex
@article{2020unifiedqa,
    title={UnifiedQA: Crossing Format Boundaries With a Single QA System},
    author={D. Khashabi and S. Min and T. Khot and A. Sabhwaral and O. Tafjord and P. Clark and H. Hajishirzi},
    journal={arXiv preprint},
    year={2020}
}
```

