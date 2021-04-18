import re
from os import listdir
from os.path import isfile, join, exists
import numpy as np

def score_string_similarity(str1, str2):
    if str1 == str2:
        return 3.0   # Better than perfect token match
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


def eval_dir(dir, checkpoints = 'all'):
    onlyfiles = [f for f in listdir(dir) if isfile(join(dir, f))]

    only_predictions = [f for f in onlyfiles if "_predictions" in f]
    if checkpoints == 'all':
        only_predictions = sorted(only_predictions)
        # print(only_predictions)
    else:
        only_predictions = [x for x in only_predictions if
                            '.csv' not in x and int(x.split('_')[-2]) > 1090500 and int(x.split('_')[-2]) < 1110000]

        if len(only_predictions) > 1:
            raise EnvironmentError


    for p in only_predictions:
        convert_predictions(input_file, dir + "/" + p, meta_file)

def convert_predictions(input, pred, meta_file):
    input_lines = []
    with open(input) as f:
        for line in f.readlines():
            input_lines.append(line.split("\t")[0])
            # print(line.split("\t")[0])

    with open(pred) as f:
        pred_lines = list(f.readlines())

    with open(meta_file) as f:
        id_lines = list(f.readlines())

    if len(pred_lines) != len(input_lines):
        print("skipping . . . ")
        return

    assert len(pred_lines) == len(input_lines), f"{len(pred_lines)} vs {len(input_lines)} / {input} / {pred}"

    outfile = open(pred + "_selected_candidate.csv", "w")

    accuracy = []
    for prediction, input, id_and_gold in zip(pred_lines, input_lines, id_lines):
        prediction = prediction.replace("\n", "").strip()
        id_and_gold_split = id_and_gold.replace("\n", "").strip().split("\t")
        id = id_and_gold_split[0]
        gold = id_and_gold_split[1]
        assert len(gold) < 3, f"gold: {gold} - id_and_gold: {id_and_gold} "
        is_numeric = False
        numeric_from_zero = False
        if len(id_and_gold_split) > 2:
            if "numeric" in id_and_gold_split[2]:
                is_numeric = True
            if "numeric_from_zero" in id_and_gold_split[2]:
                numeric_from_zero = True
        # print((is_numeric, numeric_from_zero))
        input_split = input.split("\\n")
        # print(input_split)
        candidates_string = input_split[1].strip().lower()
        candidates_split = regex.split(candidates_string)
        candidates_split = [x.strip() for x in candidates_split if len(x.strip()) > 0]
        # print(f"{prediction} <-> {candidates_split}")
        scores = [score_string_similarity(x, prediction) for x in candidates_split]
        max_idx = np.argmax(scores)
        # TODO: If multiple options has max score, look for best token alignment
        selected_ans = chr(ord('A') + max_idx)

        # print((gold, selected_ans))
        if selected_ans == gold:
            accuracy.append(1)
        else:
            accuracy.append(0)

        if is_numeric:
            # print(f"is numeric: {selected_ans} -> {chr(ord('1') + max_idx)}")
            if numeric_from_zero:
                selected_ans = chr(ord('0') + max_idx)
            else:
                selected_ans = chr(ord('1') + max_idx)

        outfile.write(f"{id},{selected_ans}\n")

        # if max(scores) == 0:
        #     print(f" ***** ERRROR: {prediction} <-> {candidates_split} ")
            # break

    print(f" *** {pred} \t {100.0 * sum(accuracy) / len(accuracy)}")


# declaring the gold labels for the target task 
input_file = "t2t-qa/t2t-data/mctest_corrected_the_separator/dev.tsv"
meta_file = "t2t-qa/t2t-data/mctest_corrected_the_separator/dev_meta.txt"

# input_file = "t2t-qa/t2t-data/race_string/dev.tsv"
# meta_file = "t2t-qa/t2t-data/race_string/dev_meta.txt"

# input_file = "t2t-qa/t2t-data/race_string/test.tsv"
# meta_file = "t2t-qa/t2t-data/race_string/test_meta.txt"

# input_file = "t2t-qa/t2t-data/openbookqa/dev.tsv"
# meta_file = "t2t-qa/t2t-data/openbookqa/dev_meta.tsv"

# input_file = "t2t-qa/t2t-data/openbookqa/test.tsv"
# meta_file = "t2t-qa/t2t-data/openbookqa/test_meta.tsv"

# input_file = "t2t-qa/t2t-data/arc_easy/dev.tsv"
# meta_file = "t2t-qa/t2t-data/arc_easy/dev_meta.tsv"

# input_file = "t2t-qa/t2t-data/arc_easy/test.tsv"
# meta_file = "t2t-qa/t2t-data/arc_easy/test_meta.tsv"

# input_file = "t2t-qa/t2t-data/arc_hard/dev.tsv"
# meta_file = "t2t-qa/t2t-data/arc_hard/dev_meta.tsv"

# input_file = "t2t-qa/t2t-data/arc_hard/test.tsv"
# meta_file = "t2t-qa/t2t-data/arc_hard/test_meta.tsv"

# input_file = "t2t-qa/t2t-data/ai2_science_elementary/test.tsv"
# meta_file = "t2t-qa/t2t-data/ai2_science_elementary/test_meta.tsv"

# input_file = "t2t-qa/t2t-data/ai2_science_middle/test.tsv"
# meta_file = "t2t-qa/t2t-data/ai2_science_middle/test_meta.tsv"

# input_file = "t2t-qa/t2t-data/qasc/dev.tsv"
# meta_file = "t2t-qa/t2t-data/qasc/dev_meta.tsv"

# input_file = "t2t-qa/t2t-data/qasc/test.tsv"
# meta_file = "t2t-qa/t2t-data/qasc/test_meta.tsv"

# input_file = "t2t-qa/t2t-data/commonsenseqa/dev.tsv"
# meta_file = "t2t-qa/t2t-data/commonsenseqa/dev_meta.tsv"

# input_file = "t2t-qa/t2t-data/commonsenseqa/test.tsv"
# meta_file = "t2t-qa/t2t-data/commonsenseqa/test_meta.tsv"

# input_file = "t2t-qa/t2t-data/physical_iqa/dev.tsv"
# meta_file = "t2t-qa/t2t-data/physical_iqa/dev_meta.tsv"

# input_file = "t2t-qa/t2t-data/physical_iqa/test.tsv"
# meta_file = "t2t-qa/t2t-data/physical_iqa/test_meta.tsv"

# input_file = "t2t-qa/t2t-data/social_iqa/dev.tsv"
# meta_file = "t2t-qa/t2t-data/social_iqa/dev_meta.tsv"

# input_file = "t2t-qa/t2t-data/social_iqa/test.tsv"
# meta_file = "t2t-qa/t2t-data/social_iqa/test_meta.tsv"


# input_file = "t2t-qa/t2t-data/winogrande_xs/dev.tsv"
# meta_file = "t2t-qa/t2t-data/winogrande_xs/dev_meta.tsv"

# input_file = "t2t-qa/t2t-data/winogrande_test/test.tsv"
# meta_file = "t2t-qa/t2t-data/winogrande_test/test_meta.tsv"


if "/qasc/" in meta_file:
    regex = re.compile("\([a-h]\)")
else:
    regex = re.compile("\([a-e]\)")

model_sizes = [
    'small',
    'base',
    'large',
    '3B',
    '11B'
]

for dir in listdir("."):
    if isfile(dir):
        continue

    for model_size in model_sizes:
        if exists(f"{dir}/{model_size}/"):
            evaluation = eval_dir(f"{dir}/{model_size}/", checkpoints='all')
