import copy
import json
import rouge
from os import listdir
from os.path import isfile, join, exists

# from rouge_score import rouge

rouge_l_evaluator = rouge.Rouge(
    metrics=["rouge-l"],
    max_n=4,
    limit_length=True,
    length_limit=100,
    length_limit_type="words",
    apply_avg=True,
    apply_best=True,
    alpha=0.5,
    weight_factor=1.2,
    stemming=True,
)

def rouge_l(p, g):
    return rouge_l_evaluator.get_scores(p, g)

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, [ground_truth])
        scores_for_ground_truths.append(score)
    if isinstance(score, dict) and "rouge-l" in score:
        max_score = copy.deepcopy(score)
        max_score["rouge-l"]["f"] = round(
            max([score["rouge-l"]["f"] for score in scores_for_ground_truths]), 2
        )
        max_score["rouge-l"]["p"] = round(
            max([score["rouge-l"]["p"] for score in scores_for_ground_truths]), 2
        )
        max_score["rouge-l"]["r"] = round(
            max([score["rouge-l"]["r"] for score in scores_for_ground_truths]), 2
        )
        return max_score
    else:
        return round(max(scores_for_ground_truths), 2)


def eval_dir(dir, checkpoints='all'):
    all_predictions = []
    onlyfiles = [f for f in listdir(dir) if isfile(join(dir, f))]
    only_targets = [f for f in onlyfiles if "_targets" in f]
    if len(only_targets) > 1:
        return None
    assert len(only_targets) == 1, f"targets: {only_targets} - dir: {dir}"

    only_inputs = [f for f in onlyfiles if "inputs" in f]
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
    if checkpoints == 'all':
        only_predictions = sorted(only_predictions)
    else:
        only_predictions = [x for x in only_predictions if
                            int(x.split('_')[-2]) > 1090500 and int(x.split('_')[-2]) < 1110000]

    for file in only_predictions:
        predictions = []
        with open(dir + file) as f:
            for l in f.readlines():
                predictions.append(l.replace("\n", "").strip())

        assert len(golds) == len(predictions), f" {len(predictions)}  / {len(golds)} "
        scores = []
        for k, v in instances.items():
            golds_subset = [golds[i] for i in v]
            rouge_l_score = metric_max_over_ground_truths(rouge_l, predictions[v[0]], golds_subset)
            scores.append(rouge_l_score["rouge-l"]["f"])

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
            evaluation = eval_dir(f"{dir}/{model_size}/", 'all')
