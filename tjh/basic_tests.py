#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 13:05:08 2021

@author: thar011
"""

from transformers import AutoTokenizer, T5ForConditionalGeneration

model_name = "allenai/unifiedqa-t5-small" # you can specify the model size here
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def run_model(input_string, **generator_args):
    input_ids = tokenizer.encode(input_string, return_tensors="pt")
    res = model.generate(input_ids, **generator_args)
    return tokenizer.batch_decode(res, skip_special_tokens=True)


run_model("which is best conductor? \\n (a) iron (b) feather")

run_model("scott filled a tray with juice and put it in a freezer. the next day, scott opened the freezer. how did the juice most likely change? \\n (a) it condensed. (b) it evaporated. (c) it became a gas. (d) it became a solid.")

run_model("which is best conductor? \\n (a) iron (b) feather (c) wood (d) plastic",
         temperature=0.9, num_return_sequences=4, num_beams=20)
