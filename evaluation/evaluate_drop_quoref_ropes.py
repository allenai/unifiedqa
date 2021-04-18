import json
import re
import string
import numpy as np
from os import listdir
from os.path import isfile, join, exists
from typing import Union, List, Tuple, Set
from scipy.optimize import linear_sum_assignment


def _answer_to_bags(
        answer: Union[str, List[str], Tuple[str, ...]]
) -> Tuple[List[str], List[Set[str]]]:
    if isinstance(answer, (list, tuple)):
        raw_spans = answer
    else:
        raw_spans = [answer]
    normalized_spans: List[str] = []
    token_bags = []
    for raw_span in raw_spans:
        normalized_span = _normalize_answer(raw_span)
        normalized_spans.append(normalized_span)
        token_bags.append(set(normalized_span.split()))
    return normalized_spans, token_bags


def _white_space_fix(text: str) -> str:
    return " ".join(text.split())


EXCLUDE = set(string.punctuation)


def _remove_punc(text: str) -> str:
    if not _is_number(text):
        return "".join(ch for ch in text if ch not in EXCLUDE)
    else:
        return text


def _lower(text: str) -> str:
    return text.lower()


def _tokenize(text: str) -> List[str]:
    return re.split(" |-", text)


def _remove_articles(text: str) -> str:
    regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    return re.sub(regex, " ", text)


def _normalize_answer(text: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    parts = [
        _white_space_fix(_remove_articles(_normalize_number(_remove_punc(_lower(token)))))
        for token in _tokenize(text)
    ]
    parts = [part for part in parts if part.strip()]
    normalized = " ".join(parts).strip()
    return normalized


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def _normalize_number(text: str) -> str:
    if _is_number(text):
        return str(float(text))
    else:
        return text


def _match_numbers_if_present(gold_bag: Set[str], predicted_bag: Set[str]) -> bool:
    gold_numbers = set()
    predicted_numbers = set()
    for word in gold_bag:
        if _is_number(word):
            gold_numbers.add(word)
    for word in predicted_bag:
        if _is_number(word):
            predicted_numbers.add(word)
    if (not gold_numbers) or gold_numbers.intersection(predicted_numbers):
        return True
    return False


def _compute_f1(predicted_bag: Set[str], gold_bag: Set[str]) -> float:
    intersection = len(gold_bag.intersection(predicted_bag))
    if not predicted_bag:
        precision = 1.0
    else:
        precision = intersection / float(len(predicted_bag))
    if not gold_bag:
        recall = 1.0
    else:
        recall = intersection / float(len(gold_bag))
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if not (precision == 0.0 and recall == 0.0)
        else 0.0
    )
    return f1


def _align_bags(predicted: List[Set[str]], gold: List[Set[str]]) -> List[float]:
    """
    Takes gold and predicted answer sets and first finds the optimal 1-1 alignment
    between them and gets maximum metric values over all the answers.
    """
    scores = np.zeros([len(gold), len(predicted)])
    for gold_index, gold_item in enumerate(gold):
        for pred_index, pred_item in enumerate(predicted):
            if _match_numbers_if_present(gold_item, pred_item):
                scores[gold_index, pred_index] = _compute_f1(pred_item, gold_item)
    row_ind, col_ind = linear_sum_assignment(-scores)

    max_scores = np.zeros([max(len(gold), len(predicted))])
    for row, column in zip(row_ind, col_ind):
        max_scores[row] = max(max_scores[row], scores[row, column])
    return max_scores


def get_metrics(
        predicted: Union[str, List[str], Tuple[str, ...]], gold: Union[str, List[str], Tuple[str, ...]]
) -> Tuple[float, float]:
    """
    Takes a predicted answer and a gold answer (that are both either a string or a list of
    strings), and returns exact match and the DROP F1 metric for the prediction.  If you are
    writing a script for evaluating objects in memory (say, the output of predictions during
    validation, or while training), this is the function you want to call, after using
    :func:`answer_json_to_strings` when reading the gold answer from the released data file.
    """
    predicted_bags = _answer_to_bags(predicted)
    gold_bags = _answer_to_bags(gold)

    if set(predicted_bags[0]) == set(gold_bags[0]) and len(predicted_bags[0]) == len(gold_bags[0]):
        exact_match = 1.0
    else:
        exact_match = 0.0

    f1_per_bag = _align_bags(predicted_bags[1], gold_bags[1])
    f1 = np.mean(f1_per_bag)
    f1 = round(f1, 2)
    return exact_match, f1


def eval_dir(dir, checkpoint='all'):
    all_predictions = []
    onlyfiles = [f for f in listdir(dir) if isfile(join(dir, f))]
    only_targets = [f for f in onlyfiles if "_targets" in f]
    if len(only_targets) > 1:
        return None
    assert len(only_targets) == 1, f"targets: {only_targets} - dir: {dir} - onlyfiles: {onlyfiles}"

    only_inputs = [f for f in onlyfiles if "input" in f]
    if len(only_inputs) > 1:
        return None
    assert len(only_inputs) == 1, f"inputs: {only_inputs} - dir: {dir}"

    instances = {}
    with open(dir + only_inputs[0]) as f:
        for i, l in enumerate(f.readlines()):
            if l not in instances:
                instances[l] = []
            instances[l].append(i)

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

    for file in only_predictions:
        predictions = []
        if ".json" in file:
            continue
        print(f" >> prediction: {file}")
        with open(dir + file) as f:
            for l in f.readlines():
                predictions.append(l.replace("\n", "").strip())

        assert len(golds) == len(predictions), f" {len(predictions)}  / {len(golds)} "
        scores = []
        for k, v in instances.items():
            golds_subset = [golds[i] for i in v]
            metric = get_metrics(predictions[v[0]], tuple(golds_subset))
            scores.append(metric[1])
            # print(metric)
            # rouge_l_score = metric_max_over_ground_truths(rouge_l, predictions[v[0]], golds_subset)
            # scores.append(rouge_l_score["rouge-l"]["f"])

        print(f" * {dir}{file} -> {100.0 * sum(scores) / len(scores)}")

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

    for model_size in model_sizes:
        if exists(f"{dir}/{model_size}/"):
            evaluation = eval_dir(f"{dir}/{model_size}/", checkpoint='all')
