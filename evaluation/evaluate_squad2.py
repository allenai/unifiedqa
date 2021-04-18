# read lines and calculate F1/EM
import collections
import string
import re
import argparse
import json
import sys

from collections import Counter
from os import listdir
from os.path import isfile, join, exists


def normalize_answer(s):
  """Lower text and remove punctuation, articles and extra whitespace."""
  def remove_articles(text):
    regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
    return re.sub(regex, ' ', text)
  def white_space_fix(text):
    return ' '.join(text.split())
  def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)
  def lower(text):
    return text.lower()
  return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
  if not s: return []
  return normalize_answer(s).split()

def compute_exact(a_gold, a_pred):
  return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
  gold_toks = get_tokens(a_gold)
  pred_toks = get_tokens(a_pred)
  common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
  num_same = sum(common.values())
  if len(gold_toks) == 0 or len(pred_toks) == 0:
    # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
    return int(gold_toks == pred_toks)
  if num_same == 0:
    return 0
  precision = 1.0 * num_same / len(pred_toks)
  recall = 1.0 * num_same / len(gold_toks)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1

def eval_dir(dir, checkpoint='all'):
    # print(dir)
    all_predictions = []
    onlyfiles = [f for f in listdir(dir) if isfile(join(dir, f))]

    gold = []
    with open(gold_all_ans) as f:
        for l in f.readlines():
            gold.append(json.loads(l.replace("\n", "")))

    only_predictions = [f for f in onlyfiles if "_predictions" in f]
    if checkpoint == 'all':
        only_predictions = sorted(only_predictions)
    else:
        only_predictions = [x for x in only_predictions if
                        int(x.split('_')[-2]) > 1090500 and int(x.split('_')[-2]) < 1110000]

    for file in only_predictions:
        print(dir + file)
        predictions = []
        with open(dir + file) as f:
            for l in f.readlines():
                predictions.append(l.replace("\n", ""))

        assert len(gold) == len(predictions), f" {len(predictions)}  / {len(gold)} "


        f1 = exact_match = total = 0
        for i, prediction in enumerate(predictions):
            # For unanswerable questions, only correct answer is empty string
            for g in gold[i]:
                if no_ans in g.lower():
                    # print(gold[i])
                    gold[i] = [""]
                    break

            if no_ans in prediction.lower():
                prediction = ""

            exact_match += max(compute_exact(a, prediction) for a in gold[i])
            f1 += max(compute_f1(a, prediction) for a in gold[i])

            total += 1

        exact_match = 100.0 * exact_match / total
        f1 = 100.0 * f1 / total
        eval = {'exact_match': exact_match, 'f1': f1}
        print(f" * {dir}{file} -> {eval}")
        all_predictions.append([file, eval])
    return all_predictions


model_sizes = [
    'small',
    'large',
    'base',
    '3B',
    '11B'
]

gold_all_ans = "t2t-data/squad2/dev_ans.jsonl"

no_ans = "no answer"

for dir in listdir("."):
    if isfile(dir):
        continue

    for model_size in model_sizes:
        if exists(f"{dir}/{model_size}/"):
            evaluation = eval_dir(f"{dir}/{model_size}/", checkpoint='all')
