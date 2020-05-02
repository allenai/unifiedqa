# UnifiedQA


You may want to check out: 
 - Our paper: TBD 
 - Our demo: https://unifiedqa.apps.allenai.org/


## Released Model Checkpoints

If you intend to create a QA system, you can use our QA-specialized models for your purpose: 

 - UnifiedQA (small) [gs://t5-data/unifiedqa/models/small](https://console.cloud.google.com/storage/browser/unifiedqa/models/small)  
 - UnifiedQA (base) [gs://t5-data/unifiedqa/models/base](https://console.cloud.google.com/storage/browser/unifiedqa/models/base)
 - UnifiedQA (large) [gs://t5-data/unifiedqa/models/large](https://console.cloud.google.com/storage/browser/unifiedqa/models/large)
 - UnifiedQA (3B) [gs://t5-data/unifiedqa/models/3b](https://console.cloud.google.com/storage/browser/unifiedqa/models/3b)
 - UnifiedQA (11B) [gs://t5-data/unifiedqa/models/11b](https://console.cloud.google.com/storage/browser/unifiedqa/models/11b)

You can use these in two ways: 
- If you don't have any training data, you can use them for [the evaluation](https://github.com/google-research/text-to-text-transfer-transformer#eval). 
- If you training data, you can use them as your initial models and [fine-tune on them](https://github.com/google-research/text-to-text-transfer-transformer#fine-tuning).

For more details see [the T5 repository](https://github.com/google-research/text-to-text-transfer-transformer). 
 
BART models will come soon! 

## How to cite

If you extend or use this work, please cite the paper: 
```
@article{2020unifiedqa,
    title={UnifiedQA: Crossing Format Boundaries With a Single QA System},
    author={D. Khashabi and T. Khot and A. Sabhwaral and O. Tafjord and P. Clark and H. Hajishirzi},
    journal={arXiv preprint},
    year={2020}
}
```

