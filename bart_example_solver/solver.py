import argparse
import json
import re
import time
import torch
import numpy as np
import urllib.request

from typing import List
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import BartTokenizer
from bart import MyBart


# Parse the input file from JSONL to a list of dictionaries.
def read_jsonl_lines(input_file: str) -> List[dict]:
    with open(input_file) as f:
        lines = f.readlines()
        return [json.loads(l.strip()) for l in lines]


def replace_punctuation(str):
    return str.replace("\"", "").replace("'", "")


# Temporary fix for bug where {}^<\` characters roundtrip into \u2047 (??) character
def fix_buggy_characters(str):
    return re.sub("[{}^\\\\`\u2047<]", " ", str)


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


class Solver(object):
    def __init__(self, checkpoint):
        self.tokenizer = BartTokenizer.from_pretrained("bart-large")
        def convert_to_single_gpu(state_dict):
            def _convert(key):
                if key.startswith('module.'):
                    return key[7:]
                return key
            return {_convert(key):value for key, value in state_dict.items()}


        file_name, headers = urllib.request.urlretrieve(checkpoint)

        self.model = MyBart.from_pretrained("bart-large", state_dict=convert_to_single_gpu(torch.load(file_name)))
        # self.model.to(torch.device("cuda"))
        self.model.eval()

        self.num_beams = 4
        self.max_input_length=512
        self.max_output_length=100


    def get_answers(self, questions, batch_size):
        '''
        :param questions: a list of string
        :param batch_size: batch_size of inference, depending on gpu availability.
                        (50 was good with one 16 GB gpu)
        '''
        questions = [x.lower() for x in questions]
        question_input = self.tokenizer.batch_encode_plus(
            ["<s> " + question.lower() for question in questions],
            pad_to_max_length=True,
            max_length=self.max_input_length)
        dataset = TensorDataset(torch.LongTensor(question_input["input_ids"]),
                                torch.LongTensor(question_input["attention_mask"]))
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
        predictions = []
        for i, batch in tqdm(enumerate(dataloader)):
            # batch = [b.to(torch.device("cuda")) for b in batch]
            outputs = self.model.generate(input_ids=batch[0],
                                          attention_mask=batch[1],
                                          num_beams=self.num_beams,
                                          min_lnegth=1,
                                          max_length=self.max_output_length,
                                          early_stopping=True, ).detach().cpu().numpy()
            for output in outputs:
                pred = self.decode(output)
                predictions.append(pred)
        return predictions

    def decode(self, tokens):
        return self.tokenizer.decode(tokens,
                                     skip_special_tokens=True,
                                     clean_up_tokenization_spaces=True).strip()


    def main(self, input_file, output_file):
        # Read the records from the test set.
        test_records = read_jsonl_lines(input_file)
        predicted_answers = []
        # Make predictions for each example in the test set.
        time_list = []
        for i, json_line in enumerate(test_records):
            print("-  -  -  -  -  -  -  -  -  -  ")
            print(f" * i: {i}")
            print(json_line)
            goal = json_line['goal'].replace("\t", " ").replace("\n", " ")
            sol1 = json_line['sol1'].replace("\t", " ").replace("\n", " ")
            sol2 = json_line['sol2'].replace("\t", " ").replace("\n", " ")
            input = f"{goal} \\n (A) {sol1} (B) {sol2}"
            print(f" * input: {input}")
            start = time.time()
            prediction = self.get_answers([input], 1)[0]
            end = time.time()
            time_list.append(end - start)
            print(f"* prediction: {prediction}")
            print(f" * avg time: {sum(time_list)/len(time_list)}")

            scores = [score_string_similarity(x, prediction) for x in [sol1, sol2]]
            max_idx = np.argmax(scores)
            print(f"max_idx: {max_idx}")
            predicted_answers.append(str(max_idx))

        # Write the predictions to the output file.
        with open(output_file, "w") as f:
            for p in predicted_answers:
                f.write(p)
                f.write("\n")
            f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='A random baseline.')
    parser.add_argument('--input-file', type=str, required=True, help='Location of test records', default=None)
    parser.add_argument('--output-file', type=str, required=True, help='Location of predictions', default=None)
    parser.add_argument('--model', type=str, required=True, help='Location of model', default=None)

    args = parser.parse_args()
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=True))
    print("=======================")
    solver = Solver(args.model)
    solver.main(args.input_file, args.output_file)
