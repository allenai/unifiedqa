# read lines and calculate F1/EM
import string
import re
import argparse
import json
import sys
from _ast import List

from collections import Counter
from os import listdir
from os.path import isfile, join, exists


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
    # print(f"{prediction} vs {ground_truth} : {normalize_answer(prediction)} : {normalize_answer(ground_truth)}")
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def eval_dir(dir,  checkpoint = 'all'):
    all_predictions = []
    onlyfiles = [f for f in listdir(dir) if isfile(join(dir, f))]
    only_targets = [f for f in onlyfiles if "_targets" in f]
    if len(only_targets) > 1:
        return None
    assert len(only_targets) == 1, f"targets: {only_targets} - dir: {dir}"
    golds = []
    with open(dir + only_targets[0]) as f:
        for l in f.readlines():
            golds.append(l.replace("\n", "").strip())

    only_predictions = [f for f in onlyfiles if "_predictions" in f]
    if checkpoint == 'all':
        only_predictions = sorted(only_predictions)
    else:
        only_predictions = [x for x in only_predictions if
                        int(x.split('_')[-2]) > 1090500 and int(x.split('_')[-2]) < 1110000]

    golds_tmp = []
    for i, id in enumerate(meta_ids):
        if id in meta_ids[:i]:
            continue
        golds_tmp.append(golds[i])
    assert len(golds) > len(golds_tmp)
    golds = golds_tmp

    for file in only_predictions:
        print(dir + file)
        predictions = []
        with open(dir + file) as f:
            for l in f.readlines():
                predictions.append(l.replace("\n", "").strip())

        preds_tmp = []
        for i, id in enumerate(meta_ids):
            if id in meta_ids[:i]:
                continue
            preds_tmp.append(predictions[i])
        assert len(predictions) > len(preds_tmp)

        predictions = preds_tmp

        assert len(golds) == len(predictions), f" {len(predictions)}  / {len(golds)} "

        f1 = exact_match = total = 0
        for i, prediction in enumerate(predictions):
            exact_match += metric_max_over_ground_truths(
                exact_match_score, prediction, [golds[i]])
            f1 += metric_max_over_ground_truths(
                f1_score, prediction, [golds[i]])
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

for dir in listdir("."):
    if isfile(dir):
        continue

    # meta_file = "t2t-data/natural_questions_direct_ans/dev_meta.tsv"
    # meta_file = "t2t-data/natural_questions_direct_ans/test_meta.tsv"
    meta_file = "t2t-qa/t2t-data/ambigqa/dev_meta.tsv"
    meta_ids = []
    with open(meta_file) as f:
        for line in f.readlines():
            meta_ids.append(line.replace("\n", ""))

    for model_size in model_sizes:
        if exists(f"{dir}/{model_size}/"):
            evaluation = eval_dir(f"{dir}/{model_size}/")
