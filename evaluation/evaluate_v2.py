#!/usr/bin/env python

"""Evaluate model predictions against target.
Usage:
   evaluate_predictions.py --eval_path=NAME --target_dataset=NAME
   evaluate_predictions.py -h| --help
Options:
    -h --help                               Show this screen
   --eval_path=NAME                         gs:// link to predictions to evaluate
"""
# example run:
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-evaluations/model:union_v2_mixture_pretrained:gs:__t5-data_pretrained_models_small_eval:race_string_mixture/small

# measuring_massive_multitask_language_understanding
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_mixture/small/test_eval --target_dataset=measuring_massive_multitask_language_understanding

# squad1_1: note that does not have a dev file
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_mixture/small/dev_eval --target_dataset=squad1_1
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_v2_mixture_pretrained:gs:__t5-data_pretrained_models_small/small/dev_eval --target_dataset=squad1_1

# squad2: note that does not have a dev file
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_mixture/small/dev_eval --target_dataset=squad2
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_v2_mixture_pretrained:gs:__t5-data_pretrained_models_small/small/dev_eval --target_dataset=squad2

# newsqa
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_mixture/small/dev_eval --target_dataset=newsqa
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_v2_mixture_pretrained:gs:__t5-data_pretrained_models_small/small/dev_eval --target_dataset=newsqa

# quoref
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_mixture/small/dev_eval --target_dataset=quoref
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_v2_mixture_pretrained:gs:__t5-data_pretrained_models_small/small/dev_eval --target_dataset=quoref

# contrast sets: quoref
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_mixture/small/dev_eval --target_dataset=contrast_sets_quoref
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_v2_mixture_pretrained:gs:__t5-data_pretrained_models_small/small/dev_eval --target_dataset=contrast_sets_quoref

# AdversarialQA: BiDAF
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_mixture/small/dev_eval --target_dataset=adversarialqa_dbidaf_dev
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_v2_mixture_pretrained:gs:__t5-data_pretrained_models_small/small/dev_eval --target_dataset=adversarialqa_dbidaf_dev

# AdversarialQA: BERT
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_mixture/small/dev_eval --target_dataset=adversarialqa_dbert_dev
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_v2_mixture_pretrained:gs:__t5-data_pretrained_models_small/small/dev_eval --target_dataset=adversarialqa_dbert_dev

# AdversarialQA: ReBerta
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_mixture/small/dev_eval --target_dataset=adversarialqa_droberta_dev
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_v2_mixture_pretrained:gs:__t5-data_pretrained_models_small/small/dev_eval --target_dataset=adversarialqa_droberta_dev

# ReCord (ex)
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_mixture/small/dev_eval --target_dataset=record_extractive
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_v2_mixture_pretrained:gs:__t5-data_pretrained_models_small/small/dev_eval --target_dataset=record_extractive

# ReCord (mc)
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_mixture/small/dev_eval --target_dataset=record_multiple_choice
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_v2_mixture_pretrained:gs:__t5-data_pretrained_models_small/small/dev_eval --target_dataset=record_multiple_choice

# RACE
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_mixture/small/dev_eval --target_dataset=race_string
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_v2_mixture_pretrained:gs:__t5-data_pretrained_models_small/small/dev_eval --target_dataset=race_string

# RACE-C
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_mixture/small/dev_eval --target_dataset=race_c
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_v2_mixture_pretrained:gs:__t5-data_pretrained_models_small/small/dev_eval --target_dataset=race_c

# OBQA (w/o IR)
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_mixture/small/dev_eval --target_dataset=openbookqa_dev
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_v2_mixture_pretrained:gs:__t5-data_pretrained_models_small/small/dev_eval --target_dataset=openbookqa_dev

# OBQA (w/ IR)
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_mixture/small/dev_eval --target_dataset=openbookqa_with_ir
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_v2_mixture_pretrained:gs:__t5-data_pretrained_models_small/small/dev_eval --target_dataset=openbookqa_with_ir

# ARC-easy (w/o IR)
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_mixture/small/dev_eval --target_dataset=arc_easy_dev
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_v2_mixture_pretrained:gs:__t5-data_pretrained_models_small/small/dev_eval --target_dataset=arc_easy_dev

# ARC-easy (w/ IR)
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_mixture/small/dev_eval --target_dataset=arc_easy_with_ir_dev
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_v2_mixture_pretrained:gs:__t5-data_pretrained_models_small/small/dev_eval --target_dataset=arc_easy_with_ir_dev

# ARC-hard (w/o IR)
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_mixture/small/dev_eval --target_dataset=arc_hard_dev
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_v2_mixture_pretrained:gs:__t5-data_pretrained_models_small/small/dev_eval --target_dataset=arc_hard_dev

# ARC-hard (w/ IR)
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_mixture/small/dev_eval --target_dataset=arc_hard_with_ir_dev
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_v2_mixture_pretrained:gs:__t5-data_pretrained_models_small/small/dev_eval --target_dataset=arc_hard_with_ir_dev

# MCTEST
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_mixture/small/dev_eval --target_dataset=mctest_corrected_the_separator
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_v2_mixture_pretrained:gs:__t5-data_pretrained_models_small/small/dev_eval --target_dataset=mctest_corrected_the_separator

# QASC (w/o IR)
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_mixture/small/dev_eval --target_dataset=qasc
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_v2_mixture_pretrained:gs:__t5-data_pretrained_models_small/small/dev_eval --target_dataset=qasc

# QASC (w/ IR)
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_mixture/small/dev_eval --target_dataset=qasc_with_ir
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_v2_mixture_pretrained:gs:__t5-data_pretrained_models_small/small/dev_eval --target_dataset=qasc_with_ir

# CommonsenseQA (dev)
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_mixture/small/dev_eval --target_dataset=commonsenseqa
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_v2_mixture_pretrained:gs:__t5-data_pretrained_models_small/small/dev_eval --target_dataset=commonsenseqa

# PhysicalIQA
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_mixture/small/dev_eval --target_dataset=physical_iqa_test
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_v2_mixture_pretrained:gs:__t5-data_pretrained_models_small/small/dev_eval --target_dataset=physical_iqa_test

# SocialIQA
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_mixture/small/dev_eval --target_dataset=social_iqa_test
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_v2_mixture_pretrained:gs:__t5-data_pretrained_models_small/small/dev_eval --target_dataset=social_iqa_test

# Winogrande
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_mixture/small/dev_eval --target_dataset=winogrande_xl
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_v2_mixture_pretrained:gs:__t5-data_pretrained_models_small/small/dev_eval --target_dataset=winogrande_xl

# HeadQA
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_mixture/small/dev_eval --target_dataset=head_qa_en_dev
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_v2_mixture_pretrained:gs:__t5-data_pretrained_models_small/small/dev_eval --target_dataset=head_qa_en_dev

# measuring_massive_multitask_language_understanding MMMLU
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_mixture/small/dev_eval --target_dataset=measuring_massive_multitask_language_understanding
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_v2_mixture_pretrained:gs:__t5-data_pretrained_models_small/small/dev_eval --target_dataset=measuring_massive_multitask_language_understanding

# AQUA-RAT
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_mixture/small/dev_eval --target_dataset=aqua_rat_test
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_v2_mixture_pretrained:gs:__t5-data_pretrained_models_small/small/dev_eval --target_dataset=aqua_rat_test

# ReClor
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_mixture/small/dev_eval --target_dataset=reclor
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_v2_mixture_pretrained:gs:__t5-data_pretrained_models_small/small/dev_eval --target_dataset=reclor

# QUAIL
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_mixture/small/dev_eval --target_dataset=quail
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_v2_mixture_pretrained:gs:__t5-data_pretrained_models_small/small/dev_eval --target_dataset=quail

# OneStop: elementary
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_mixture/small/test_eval --target_dataset=onestopqa_elementry
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_v2_mixture_pretrained:gs:__t5-data_pretrained_models_small/small/test_eval --target_dataset=onestopqa_elementry

# OneStop: intermediate
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_mixture/small/test_eval --target_dataset=onestopqa_intermediate
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_v2_mixture_pretrained:gs:__t5-data_pretrained_models_small/small/test_eval --target_dataset=onestopqa_intermediate

# OneStop: advanced
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_mixture/small/test_eval --target_dataset=onestopqa_advanced
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_v2_mixture_pretrained:gs:__t5-data_pretrained_models_small/small/test_eval --target_dataset=onestopqa_advanced

# MCScript
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_mixture/small/dev_eval --target_dataset=mcscript
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_v2_mixture_pretrained:gs:__t5-data_pretrained_models_small/small/dev_eval --target_dataset=mcscript

# MCScript2
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_mixture/small/dev_eval --target_dataset=mcscript2
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_v2_mixture_pretrained:gs:__t5-data_pretrained_models_small/small/dev_eval --target_dataset=mcscript2

# Cosmosqa
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_mixture/small/dev_eval --target_dataset=cosmosqa
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_v2_mixture_pretrained:gs:__t5-data_pretrained_models_small/small/dev_eval --target_dataset=cosmosqa

# ProcessBank
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_mixture/small/test_eval --target_dataset=processbank_test
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_v2_mixture_pretrained:gs:__t5-data_pretrained_models_small/small/test_eval --target_dataset=processbank_test

# DREAM
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_mixture/small/dev_eval --target_dataset=dream
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_v2_mixture_pretrained:gs:__t5-data_pretrained_models_small/small/dev_eval --target_dataset=dream

# PROST (w/o context)
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_mixture/small/test_eval --target_dataset=prost_multiple_choice_with_no_context
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_v2_mixture_pretrained:gs:__t5-data_pretrained_models_small/small/test_eval --target_dataset=prost_multiple_choice_with_no_context

# PROST (w/ context)
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_mixture/small/test_eval --target_dataset=prost_multiple_choice_with_context
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_v2_mixture_pretrained:gs:__t5-data_pretrained_models_small/small/test_eval --target_dataset=prost_multiple_choice_with_context

# boolq
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_mixture/small/dev_eval --target_dataset=boolq_mixture
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_v2_mixture_pretrained:gs:__t5-data_pretrained_models_small/small/dev_eval --target_dataset=boolq_mixture

# boolq_np
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_mixture/small/dev_eval --target_dataset=boolq_np_mixture
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_v2_mixture_pretrained:gs:__t5-data_pretrained_models_small/small/dev_eval --target_dataset=boolq_np_mixture

# boolq contrast set
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_mixture/small/dev_eval --target_dataset=contrast_sets_boolq
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_v2_mixture_pretrained:gs:__t5-data_pretrained_models_small/small/dev_eval --target_dataset=contrast_sets_boolq

# StrategyQA
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_mixture/small/dev_eval --target_dataset=strategyqa
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_v2_mixture_pretrained:gs:__t5-data_pretrained_models_small/small/dev_eval --target_dataset=strategyqa

# PubmedQA
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_mixture/small/test_eval --target_dataset=pubmedqa_pqal_short_ans
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_v2_mixture_pretrained:gs:__t5-data_pretrained_models_small/small/test_eval --target_dataset=pubmedqa_pqal_short_ans

# CommonsenseQA 2.0: note, it's a yes/no dataset
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_mixture/small/dev_eval --target_dataset=csqa2
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_v2_mixture_pretrained:gs:__t5-data_pretrained_models_small/small/dev_eval --target_dataset=csqa2

# NarrativeQA
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_mixture/small/dev_eval --target_dataset=narrativeqa
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_v2_mixture_pretrained:gs:__t5-data_pretrained_models_small/small/dev_eval --target_dataset=narrativeqa

# ROPES
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_mixture/small/dev_eval --target_dataset=ropes
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_v2_mixture_pretrained:gs:__t5-data_pretrained_models_small/small/dev_eval --target_dataset=ropes

# qaconv
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_mixture/small/test_eval --target_dataset=qaconv
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_v2_mixture_pretrained:gs:__t5-data_pretrained_models_small/small/test_eval --target_dataset=qaconv

# TweetQA
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_mixture/small/dev_eval --target_dataset=tweetqa
# python evaluate_v2.py  --eval_path=gs://danielk-files/t5-models/union_v2_mixture_pretrained:gs:__t5-data_pretrained_models_small/small/dev_eval --target_dataset=tweetqa



import json
import random
import re
import string
from typing import List
import t5
from collections import Counter
import tensorflow.compat.v1 as tf
from docopt import docopt
from google.cloud import storage
from pathlib import Path
import re
from tweetqa_eval import ans_score as tweetqa_metric
from evaluate_drop_quoref_ropes import f1_metrics as drop_quoref_ropes_f1_metrics
from evaluate_squad2 import compute_f1 as squad2_compute_f1
from qaconv_eval import compute_f1 as qaconv_compute_f1
from qaconv_eval import normalize_answer as qaconv_normalize_answer
from qaconv_eval import add_word_number_mapping as qaconv_add_word_number_mapping
from evaluate_narrativeqa import rouge_l
from os import listdir
from os.path import isfile, join, exists
import numpy as np
from datasets import load_dataset

def get_lines_from_file(bucket_name, file_name):
    full_file_name = f'gs://{bucket_name}/{file_name}'
    lines = []
    with tf.io.gfile.GFile(full_file_name) as ip_lines:
        for line in ip_lines:
            lines.append(line.strip())
    return lines

def score_string_similarity(str1, str2):
    if str1 == str2:
        return 3.0  # Better than perfect token match
    str1 = fix_buggy_characters(replace_punctuation(str1))
    str2 = fix_buggy_characters(replace_punctuation(str2))
    if str1 == str2:
        return 2.0
    if " " in str1 or " " in str2:
        str1_split = str1.split(" ")
        str2_split = str2.split(" ")
        overlap = list(set(str1_split) & set(str2_split))
        return len(overlap) / max(len(str1_split), len(str2_split))
    else:
        if str1 == str2:
            return 1.0
        else:
            return 0.0


def replace_punctuation(str):
    return str.replace("\"", "").replace("'", "")


# Temporary fix for bug where {}^<\` characters roundtrip into \u2047 (??) character
def fix_buggy_characters(str):
    return re.sub("[{}^\\\\`\u2047<]", " ", str)

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction.lower(), ground_truth.lower())
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate_generic(target_lines: List, pred_lines: List, variant_type):
    if type(target_lines[0]) == str:
        # strip and turn into *list of* strings
        golds = [[x.replace("\n", "").strip()] for x in target_lines]
    else:
        golds = target_lines
    predictions = [x.replace("\n", "").strip() for x in pred_lines]

    assert len(golds) == len(predictions), f" {len(predictions)}  / {len(golds)} "

    score = total = 0
    for i, prediction in enumerate(predictions):
        if variant_type == "generic-f1":
            out = metric_max_over_ground_truths(f1_score, prediction, golds[i])
            score += out
        elif variant_type == "squad2-f1":
            no_ans = "no answer"
            for g in golds[i]:
                if no_ans in g.lower():
                    golds[i] = [""]
                    break
            if no_ans in prediction.lower():
                prediction = ""
            score += metric_max_over_ground_truths(squad2_compute_f1, prediction, golds[i])
        elif variant_type == "tweetqa-textgen":
            out = tweetqa_metric(prediction, golds[i])
            score += out["bleu"]
        elif variant_type == "allennlp-f1":
            out = drop_quoref_ropes_f1_metrics(prediction, tuple(golds[i]))
            score += out[1]
        elif variant_type == "qaconv-f1":
            prediction = qaconv_normalize_answer(prediction)
            gold_answers = golds[i]
            gold_answers = [qaconv_normalize_answer(a) for a in gold_answers]
            gold_answers += qaconv_add_word_number_mapping(gold_answers)
            out = metric_max_over_ground_truths(qaconv_compute_f1, prediction, gold_answers)
            score += out
        elif variant_type == "rouge-l":
            rouge_l_score = metric_max_over_ground_truths(rouge_l, prediction, golds[i])
            score += rouge_l_score["rouge-l"]["f"]
        total += 1

    score = 100.0 * score / total
    
    return score


def evaluate_multple_choice_predictions(target_lines, pred_lines, meta_lines, regex, outfile):
    accuracy = []
    for prediction, input, id_and_gold in zip(pred_lines, target_lines, meta_lines):
        prediction = prediction.replace("\n", "").strip().lower()
        id_and_gold_split = id_and_gold.replace("\n", "").strip().split("\t")
        id = id_and_gold_split[0]
        gold = id_and_gold_split[1].strip()
        assert len(gold) < 3, f"gold: {gold} - id_and_gold: {id_and_gold} "
        is_numeric = False
        numeric_from_zero = False
        if len(id_and_gold_split) > 2:
            if "numeric" in id_and_gold_split[2]:
                is_numeric = True
            if "numeric_from_zero" in id_and_gold_split[2]:
                numeric_from_zero = True
        input_split = input.split("\\n")
        candidates_string = input_split[1].strip().lower()
        candidates_split = regex.split(candidates_string)
        candidates_split = [x.strip() for x in candidates_split if len(x.strip()) > 0]
        scores = [score_string_similarity(x, prediction) for x in candidates_split]
        max_idx = np.argmax(scores)
        # TODO: If multiple options has max score, look for best token alignment
        selected_ans = chr(ord('A') + max_idx)
        if selected_ans == gold:
            accuracy.append(1)
        else:
            accuracy.append(0)

        if is_numeric:
            if numeric_from_zero:
                selected_ans = chr(ord('0') + max_idx)
            else:
                selected_ans = chr(ord('1') + max_idx)

        outfile.write(f"{id},{selected_ans}\n")

    acc = 100.0 * sum(accuracy) / len(accuracy)
    return acc


def create_map(list1, list2):
    new_dict = {}
    for (key, value) in zip(list1, list2):
        if key in new_dict:
            new_dict[key].append(value)
        else:
            new_dict[key] = [value]
    return new_dict


def evaluate_for_size(eval_path, target_dataset):

    path_regex = r'gs://(?P<bucket_name>[^/]+)/(?P<file_path>.+)'
    m = re.match(path_regex, eval_path)
    bucket_name = m.groupdict()['bucket_name']
    path = m.groupdict()['file_path']

    print(f'Bucket name: {bucket_name}, Path: {path}')
    storage_client = storage.Client()
    blobs = list(storage_client.list_blobs(
        bucket_name, prefix=path
    ))

    eval_type = ""
    # for multple-choice, use the reference labels
    if "squad1_1" in target_dataset:
        eval_type = "generic-f1"
        targets = []
        gold_all_ans = "t2t-qa/t2t-data/squad1_1/dev_ans.jsonl"
        with open(gold_all_ans) as f:
            for l in f.readlines():
                targets.append(json.loads(l.replace("\n", "")))
    elif "record_extractive" in target_dataset:
        eval_type = "generic-f1"
        if "dev" in eval_path:
            split = 'dev'
        else:
            split = 'test' # the test set does not have any gold labels
        tsv_file = get_lines_from_file(bucket_name=bucket_name, file_name = f"data/record_extractive/{split}_meta.txt")
        targets = [target_line for target_line in tsv_file]
        targets = [x.split("\t")[1].split("//") for x in targets]
    elif "narrativeqa" in target_dataset:
        eval_type = "rouge-l"
        targets_file = ([blob for blob in blobs if blob.name.endswith('_targets') and
                         "/" + target_dataset + "_mixture" in blob.name])[0]
        print(f" * targets_file.name: {targets_file.name}")
        targets = get_lines_from_file(bucket_name, targets_file.name)
        targets = [x.split("\t")[0] for x in targets]
        # targets = [x.split("///") for x in targets]
    elif "tweetqa" in target_dataset:
        eval_type = "tweetqa-textgen"
        if "dev" in eval_path:
            gold_all_ans = f"t2t-qa/t2t-data/tweetqa/dev_meta.tsv"
        else:
            gold_all_ans = f"t2t-qa/t2t-data/tweetqa/test_meta.tsv"
        targets = []
        with open(gold_all_ans) as f:
            for l in f.readlines():
                id_and_gold_split = l.replace("\n", "").strip().split("\t")
                # id = id_and_gold_split[0]
                gold = id_and_gold_split[1].strip()
                targets.append(gold.split("///"))
    elif "squad2" in target_dataset:
        eval_type = "squad2-f1"
        targets = []
        gold_all_ans = "t2t-qa/t2t-data/squad2/dev_ans.jsonl"
        with open(gold_all_ans) as f:
            for l in f.readlines():
                targets.append(json.loads(l.replace("\n", "")))
    elif len([x for x in ["quoref", "drop", "ropes"] if x in target_dataset]) > 0:
        eval_type = "allennlp-f1"
        targets_file = ([blob for blob in blobs if blob.name.endswith('_targets') and
                         "/" + target_dataset + "_mixture" in blob.name])[0]
        print(f" * targets_file.name: {targets_file.name}")
        targets = get_lines_from_file(bucket_name, targets_file.name)
        targets = [x.split("\t")[0] for x in targets]
        targets = [x.split("///") for x in targets]
    elif "adversarialqa_" in target_dataset:
        eval_type = "generic-f1"
        if 'dbidaf' in target_dataset:
            dataset = load_dataset("adversarial_qa", 'dbidaf')
        elif 'dbert' in target_dataset:
            dataset = load_dataset("adversarial_qa", 'dbert')
        elif 'droberta' in target_dataset:
            dataset = load_dataset("adversarial_qa", 'droberta')
        else:
            raise Exception("unknown split for adversarialqa")
        if "dev" in eval_path:
            split = 'validation'
        else:
            split = 'test' # the test set does not have any gold labels
        targets = []
        for x in dataset[split]:
            targets.append([a.lower().replace("\t", " ").replace("\n", " ") for a in x['answers']['text']])
    elif "qaconv" in target_dataset:
        eval_type = "qaconv-f1"
        if "dev" in eval_path:
            split = 'dev'
        else:
            split = 'test' # the test set does not have any gold labels
        tsv_file = get_lines_from_file(bucket_name=bucket_name, file_name = f"data/qaconv/{split}_meta.txt")
        targets = [target_line for target_line in tsv_file]
        targets = [x.split("\t")[1].split("//") for x in targets]
    elif len([x for x in ["boolq", "csqa2", "squad", "multirc", "newsqa", "strategyqa", "pubmedqa_pqal_short_ans"] if x in target_dataset]) > 0:
        eval_type = "generic-f1"
        targets_file = ([blob for blob in blobs if blob.name.endswith('_targets') and target_dataset in blob.name])[0]
        print(f" * targets_file.name: {targets_file.name}")
        targets = get_lines_from_file(bucket_name, targets_file.name)
        targets = [target_line for target_line in targets]
        targets = [x.split("\t")[0] for x in targets]
    elif len([x for x in ["race_string", "race_c", "openbookqa", "aqua_rat", "quail",
                          "mcscript", "mcscript2", "measuring_massive_multitask_language_understanding",
                          "record_multiple_choice", "qasc", "qasc_with_ir", "mctest_corrected_the_separator",
                          "arc_hard_dev", "arc_hard_with_ir_dev", "arc_easy_with_ir_dev", "arc_easy_dev",
                          "openbookqa_dev", "openbookqa_with_ir_dev", "commonsenseqa", "dream",
                          "prost_multiple_choice_with_no_context", "prost_multiple_choice_with_context",
                          "head_qa_en_dev", "reclor", "cosmosqa",
                          "onestopqa_advanced", "onestopqa_elementry", "onestopqa_intermediate",
                          "processbank_test", "winogrande_xl", "social_iqa", "social_iqa_test",
                          "physical_iqa", "physical_iqa_test"] if x in target_dataset]) > 0:
        eval_type = "multiple-choice"
        print(f" * target_dataset: {target_dataset}")
        if "dev_eval" in eval_path:
            input_file = f"t2t-data/{target_dataset}/dev.tsv"
            meta_file = f"t2t-data/{target_dataset}/dev_meta.tsv"
        else:
            input_file = f"t2t-qa/t2t-data/{target_dataset}/test.tsv"
            meta_file = f"t2t-data/{target_dataset}/test_meta.tsv"

        with open(input_file) as f:
            inputs = [x.split("\t")[0] for x in f.readlines()]

        if "/qasc/" in meta_file:
            regex = re.compile("\([a-h]\)")
        else:
            regex = re.compile("\([a-e]\)")

        with open(meta_file) as f:
            meta_lines = list(f.readlines())
    else:
        raise Exception("unknown dataset . . . ")

    prediction_checkpoints = [blob for blob in blobs if blob.name.endswith('_predictions')]
    best_score = 0.0
    best_checkpoint = None
    for prediction_checkpoint in prediction_checkpoints:
        if "/" + target_dataset not in prediction_checkpoint.name :
            continue
        print(f" * eval_type: {eval_type}")
        print(f' * Evaluating prediction checkpoint {prediction_checkpoint.name}')
        predictions = get_lines_from_file(bucket_name, prediction_checkpoint.name)
        predictions = [x.split("\t")[0] for x in predictions]
        if eval_type == "multiple-choice":
            if len(inputs) != len(predictions):
                print(f' 🛑 🛑 🛑 🛑 Something is wrong! The no. of predictions {len(predictions)} does not match no. of target labels {len(inputs)}.')
                continue
            if  len(meta_lines) != len(predictions):
                print(f' 🛑 🛑 🛑 🛑 Something is wrong! {len(predictions)} vs {len(meta_lines)} ')
                continue
            # create directories, if not availeble (Python >=3.5)
            Path(prediction_checkpoint.name).mkdir(parents=True, exist_ok=True)
            outfile = open(prediction_checkpoint.name + ".csv", "w")
            score = evaluate_multple_choice_predictions(inputs, predictions, meta_lines, regex, outfile)
        elif eval_type in ["generic-f1", "squad2-f1", "qaconv-f1", "allennlp-f1", "tweetqa-textgen", "rouge-l"]:
            if len(targets) != len(predictions):
                print(f' 🛑 🛑 🛑 🛑 Something is wrong! The no. of predictions {len(predictions)} does not match no. of target labels {len(targets)}.')
                continue
            score = evaluate_generic(targets, predictions, eval_type)
        else:
            raise Exception(f" * Unknown eval type: {eval_type}")

        print(f'Score on current checkpoint: {score}')
        if score > best_score:
            best_score = score
            best_checkpoint = prediction_checkpoint.name
    print(f' * Best checkpoint: {best_checkpoint}. Best score: {best_score}')

if __name__ == "__main__":
    # parse command line arguments
    args = docopt(__doc__)
    eval_path = args["--eval_path"]
    target_dataset = args["--target_dataset"]
    # eval_metric = args["--eval_metric"]
    model_sizes = [
        'small',
        'base',
        'large',
        '3B',
        '11B'
    ]
    for size in model_sizes:
        print(f" * * * * * * {size} * * * * * * ")
        new_eval_path = eval_path.replace("small", size)
        evaluate_for_size(new_eval_path, target_dataset)
