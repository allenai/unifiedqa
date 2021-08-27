#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 13:05:08 2021

@author: thar011

UnifiedQA initial tests:
    T5 checkpoints tests
    followed by BART tests    

"""

# UnifiedQA T5 checkpoints tests:
    
from transformers import AutoTokenizer, T5ForConditionalGeneration

# inference on <= 3B models work
#model_name = "allenai/unifiedqa-t5-small" # you can specify the model size here
#model_name = "allenai/unifiedqa-t5-base" # you can specify the model size here
model_name = "allenai/unifiedqa-t5-large" # you can specify the model size here
#model_name = "allenai/unifiedqa-t5-3b" # you can specify the model size here
model_name = "allenai/unifiedqa-t5-11b" # you can specify the model size here
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def run_model(input_string, **generator_args):
    input_ids = tokenizer.encode(input_string, return_tensors="pt")
    res = model.generate(input_ids, **generator_args)
    return tokenizer.batch_decode(res, skip_special_tokens=True)


run_model("which is best conductor? \\n (a) iron (b) feather")  #['iron']
run_model("which is best conductor? \\n ") # ['no answer>']
run_model("which is best conductor - iron or feather? \\n ") # ['iron']
run_model("Name a conductor of electricty? \\n Name any conductor") # ['any conductor']
run_model("Name a conductor of electricty? \\n ") # ['yes']
run_model("Name a conductor of electricity: \\n ") # ['yes']
run_model("What is 53 + 9521? \\n ") # ['no answer>']



run_model("scott filled a tray with juice and put it in a freezer. the next day, scott opened the freezer. how did the juice most likely change? \\n (a) it condensed. (b) it evaporated. (c) it became a gas. (d) it became a solid.")

run_model("which is best conductor? \\n (a) iron (b) feather (c) wood (d) plastic",
         temperature=0.9, num_return_sequences=4, num_beams=20)


# BART tests (run from  unifiedqa-tjh/bart directory):
    
import torch
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
from bart import MyBart

base_model = "facebook/bart-large"
#unifiedqa_path = "unifiedQA-uncased/best-model.pt" # path to the downloaded checkpoint
unifiedqa_path = "/data/thar011/ckpts/unifiedqa-bart-large-allenai/unifiedQA-uncased/best-model.pt" # path to the downloaded checkpoint

tokenizer = BartTokenizer.from_pretrained(base_model)
model = MyBart.from_pretrained(base_model, state_dict=torch.load(unifiedqa_path))
model.eval()

# ERROR: TypeError: forward() got an unexpected keyword argument 'past_key_values'
x = model.generate_from_string("Which is best conductor? \\n (A) iron (B) feather", tokenizer=tokenizer)
print(x)

x = model.generate_from_string("What is the sum of 3 and 5? \\n (A) 8 (B) 3 (C) 5 (D) 10", tokenizer=tokenizer)
print(x)


#try basic bart model (no error):    
model = BartForConditionalGeneration.from_pretrained(base_model)
model.eval()
run_model("which is best conductor? \\n (a) iron (b) feather") #['whichwhich is best conductor?']




