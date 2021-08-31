import itertools
import json
import os
import csv
import errno
import random
from random import shuffle
from typing import List
import spacy
from tqdm import tqdm
import pandas as pd
import codecs
import nltk
import glob
import xml.etree.ElementTree as ET
from datasets import load_dataset
import statistic

nltk.download("stopwords")
from nltk.corpus import stopwords

STOPWORDS = stopwords.words("english")
STOPWORDS = [stopword + " " for stopword in STOPWORDS]
nlp = spacy.load("en_core_web_sm")


def commonsenseqa():
    def read_file(file, split):
        fout = open(f"commonsenseqa/{split}.tsv", "w")
        fout_meta = open(f"commonsenseqa/{split}_meta.tsv", "w")
        with open(file) as f:
            for line in f.readlines():
                json_line = json.loads(line)

                candidates_str = " ".join([f"({x['label']}) {x['text']}" for x in json_line['question']['choices']])
                if split != "test":
                    selected_ans_string = [x['text'] for x in json_line['question']['choices'] if
                                           json_line['answerKey'] == x['label']]
                    assert len(selected_ans_string) == 1, f"{len(selected_ans_string)} -- {json_line['answerKey']}"
                json_line['question']['stem'] = json_line['question']['stem'].replace("\t", " ").replace("\n", "")
                candidates_str = candidates_str.replace("\t", " ").replace("\n", "")

                if split == "test":
                    fout_meta.write(f"{json_line['id']}\t-\n")
                    fout.write(f"{json_line['question']['stem']} \\n {candidates_str}\t-\n")
                else:
                    fout_meta.write(f"{json_line['id']}\t{json_line['answerKey']}\n")
                    selected_ans_string[0] = selected_ans_string[0].replace("\t", " ").replace("\n", "")
                    fout.write(f"{json_line['question']['stem']} \\n {candidates_str}\t{selected_ans_string[0]}\n")

    read_file("commonsenseqa/dev_rand_split.jsonl", "dev")
    read_file("commonsenseqa/train_rand_split.jsonl", "train")
    read_file("commonsenseqa/test_rand_split_no_answers.jsonl", "test")


def read_qas_paragraphs(file):
    map = {}
    length_list = []
    with open(file) as f:
        for line in f.readlines():
            json_line = json.loads(line)
            sentence_list = []
            for c in json_line['question']['choices']:
                doc = nlp(c['para'])
                all_sentences = [sent.text.strip() for sent in doc.sents]
                sentence_list += all_sentences[-4:]
            sentence_list = list(set(sentence_list))
            map[json_line['id']] = " ".join(sentence_list)

            length_list.append(len(map[json_line['id']].split(" ")))
    print(length_list)
    return map


def qasc():
    qasc_para = {}
    map1 = read_qas_paragraphs("QASC_Dataset_2Step/train.jsonl")
    map2 = read_qas_paragraphs("QASC_Dataset_2Step/test.jsonl")
    map3 = read_qas_paragraphs("QASC_Dataset_2Step/dev.jsonl")
    qasc_para.update(map1)
    qasc_para.update(map2)
    qasc_para.update(map3)

    def process_file(file, split, with_para):
        outdir = "qasc"
        if with_para:
            outdir = "qasc_with_ir"
        fout = open(f"{outdir}/{split}.tsv", "w")
        fout_meta = open(f"{outdir}/{split}_meta.tsv", "w")
        with open(file) as f:
            for line in f.readlines():
                json_line = json.loads(line)
                para = ""
                if with_para:
                    para = "\\n" + qasc_para[json_line['id']].replace("\n", " ").replace("\t", " ")
                candidates_str = " ".join([f"({x['label']}) {x['text']}" for x in json_line['question']['choices']])
                if 'answerKey' in json_line:
                    selected_ans_string = [x['text'] for x in json_line['question']['choices'] if
                                           json_line['answerKey'] == x['label']]
                    assert len(selected_ans_string) == 1, f"{len(selected_ans_string)} -- {json_line['answerKey']}"
                    ansKey = json_line['answerKey']
                else:
                    selected_ans_string = ['-']
                    ansKey = '-'
                fout.write(f"{json_line['question']['stem']} \\n {candidates_str}{para}\t{selected_ans_string[0]}\n")
                fout_meta.write(f"{json_line['id']}\t{ansKey}\n")

    for with_para in [True, False]:
        process_file("QASC_Dataset/dev.jsonl", "dev", with_para)
        process_file("QASC_Dataset/test.jsonl", "test", with_para)
        process_file("QASC_Dataset/train.jsonl", "train", with_para)


def boolq_contrast_sets():
    def read_file(split):
        fout = open(f"contrast_sets_boolq/{split}.tsv", "w")
        # fout_meta = open(f"boolq-experts/{split}_meta.tsv", "w")
        with open("contrast_sets/boolq_expert_perturbations.json") as f:
            json_content = json.load(f)
            for entry in json_content['data']:
                passage = f"({entry['title']}) {entry['paragraph']}"
                for q in entry['perturbed_questions']:
                    if '?' not in q['perturbed_q']:
                        q['perturbed_q'] += '?'
                    if q['answer'] == 'TRUE':
                        ans = "yes"
                    else:
                        ans = "no"
                    fout.write(f"{q['perturbed_q']} \\n {passage}\t{ans}\n")

    read_file("train")
    read_file("test")

def physical_iqa():
    def read_file(split):
        fout = open(f"physical_iqa/{split}.tsv", "w")
        fout_meta = open(f"physical_iqa/{split}_meta.tsv", "w")

        with open(f"physicaliqa-train-dev/{split}-labels.lst") as f:
            labels = [line.replace("\n", "").strip() for line in f.readlines()]

        counter = 0
        with open(f"physicaliqa-train-dev/{split}.jsonl") as f:
            for idx, line in enumerate(f.readlines()):
                label = labels[idx]
                json_line = json.loads(line)
                id = json_line['id']
                goal = json_line['goal'].replace("\t", " ").replace("\n", " ")
                sol1 = json_line['sol1'].replace("\t", " ").replace("\n", " ")
                sol2 = json_line['sol2'].replace("\t", " ").replace("\n", " ")
                assert label == "1" or label == "0", f" * label: {label}"
                ans = sol1
                ans_label_ab = "A"
                if label == "1":
                    ans = sol2
                    ans_label_ab = "B"
                ans = ans.replace("\t", " ").replace("\n", " ")
                fout.write(f"{goal} \\n (A) {sol1} (B) {sol2} \t {ans}\n")
                fout_meta.write(f"{id}\t{ans_label_ab}\t numeric_from_zero \t{ans}\n")
                counter += 1
        return counter

    dev_count = read_file("dev")
    train_count = read_file("train")
    with open(f"physical_iqa/counts.json", "w+") as outfile:
        json.dump({"train": train_count, "dev": dev_count}, outfile)

def social_iqa():
    def read_file(split):
        fout = open(f"social_iqa/{split}.tsv", "w")
        fout_meta = open(f"social_iqa/{split}_meta.tsv", "w")

        with open(f"socialiqa-train-dev/{split}-labels.lst") as f:
            labels = [line.replace("\n", "").strip() for line in f.readlines()]

        counter = 0
        with open(f"socialiqa-train-dev/{split}.jsonl") as f:
            for idx, line in enumerate(f.readlines()):
                label = labels[idx]
                json_line = json.loads(line)
                context = json_line['context'].replace("\t", " ").replace("\n", " ")
                question = json_line['question'].replace("\t", " ").replace("\n", " ")
                answerA = json_line['answerA'].replace("\t", " ").replace("\n", " ")
                answerB = json_line['answerB'].replace("\t", " ").replace("\n", " ")
                answerC = json_line['answerC'].replace("\t", " ").replace("\n", " ")
                assert label == "1" or label == "2" or label == "3", f" * label: {label}"
                ans = answerA
                abc_label = "A"
                if label == "2":
                    ans = answerB
                    abc_label = "B"
                if label == "3":
                    ans = answerC
                    abc_label = "C"
                ans = ans.replace("\t", " ").replace("\n", " ")
                fout.write(f"{question} \\n (A) {answerA} (B) {answerB} (C) {answerC} \\n {context} \t {ans}\n")
                fout_meta.write(f"-\t{abc_label}\t numeric \t{ans} \n")
                counter += 1

        return counter


    dev_count = read_file("dev")
    train_count = read_file("train")
    with open(f"social_iqa/counts.json", "w+") as outfile:
        json.dump({"train": train_count, "dev": dev_count}, outfile)


def drop_contrast_sets():
    def read_file(split):
        fout = open(f"contrast_sets_drop/{split}.tsv", "w")
        fout_meta = open(f"contrast_sets_drop/{split}_meta.tsv", "w")
        with open("drop_dataset/DROP/drop_contrast_sets_test.json") as f:
            json_content = json.load(f)
            for title, content in json_content.items():
                for qp in content['qa_pairs']:
                    answer = qp['answer']
                    number = answer['number']
                    spans = answer['spans']
                    if len(spans) > 0:
                        ans_text = ", ".join(spans)
                    elif len(number) > 0:
                        ans_text = number
                    else:
                        day = answer['date']['day']
                        month = answer['date']['month']
                        year = answer['date']['year']
                        if len(month) > 0:
                            ans_text += month
                        if len(day) > 0:
                            ans_text += f" {day}"
                        if len(year) > 0:
                            ans_text += f" {year}"

                    # assert ans_text != ""
                    # print(ans_text)
                    if ans_text == "":
                        print(" >>>> skipping the question . . . ")
                        continue

                    fout.write(f"{qp['question']} \\n {content['passage']}\t{ans_text}\n")
                    fout_meta.write(f"{qp['query_id']}\n")

    read_file("train")
    read_file("test")


def quoref_contrast_sets():
    def read_file(split):
        fout = open(f"contrast_sets_quoref/{split}.tsv", "w")
        fout_meta = open(f"contrast_sets_quoref/{split}_meta.tsv", "w")
        with open(
                "drop_dataset/quoref/quoref_test_perturbations_20191206_merged.json") as f:
            json_content = json.load(f)
            for entry in json_content['data']:
                entry['title'] = entry['title'].replace("\n", " ").replace("\t", " ")
                for p in entry['paragraphs']:
                    p['context'] = p['context'].replace("\n", " ").replace("\t", " ")
                    passage = f"({entry['title']}) {p['context']}"
                    for q in p['qas']:
                        answers = "///".join([x['text'] for x in q['answers']])
                        fout.write(f"{q['question']}\\n{passage}\t{answers}\n")
                        fout_meta.write(f"{q['id']}\n")

    read_file("train")
    read_file("test")


def ropes_contrast_sets():
    def read_file(split):
        fout = open(f"contrast_sets_ropes/{split}.tsv", "w")
        fout_meta = open(f"contrast_sets_ropes/{split}_meta.tsv", "w")
        with open(
                "drop_dataset/ropes/data/ropes_contrast_set_032820.json") as f:
            json_content = json.load(f)
            for para in json_content['data'][0]['paragraphs']:
                context = f"{para['background']} {para['situation']}".replace("\n", " ").replace("\t", " ")
                for qa in para['qas']:
                    question = qa['question'].replace("\n", " ").replace("\t", " ")
                    for a in qa['answers']:
                        answer = a['text'].replace("\n", " ").replace("\t", " ")
                        fout.write(f"{question} \\n {context}\t{answer}\n")
                        fout_meta.write(f"{qa['id']}\n")

    read_file("train")
    read_file("test")


def mctest():
    def read_and_convert_mctest_data(file, output_dir, out_file, write_format="w+"):
        # out_file = file.split("/")[-1].replace(".tsv", "")
        # fdataset_idx = open(f"{output_dir}/{out_file}_idx.tsv", "w+")
        fdataset_string = open(f"{output_dir}/{out_file}.tsv", write_format)
        fmeta = open(f"{output_dir}/{out_file}_meta.txt", write_format)
        global all_inputs
        all_inputs = []
        all_answers = []
        all_meta = []
        all_candidates = []
        with open(file) as f:
            for l in f.readlines():
                line_split = l.replace("\n", "").replace("\\newline", " ").split("\t")
                pid = line_split[0]
                paragraph = line_split[2]

                def get_question_and_candidates(split_row: List[str]):
                    kind = "one" if "one: " in split_row[0] else "multiple"
                    question = split_row[0].replace("one: ", "").replace("multiple: ", "")
                    candidates = split_row[1:5]
                    all_candidates.append(candidates)
                    candidates = " ".join([f"({chr(ord('A') + i)}) {x}" for i, x in enumerate(candidates)])
                    # fmeta.write(pid + "\t" + kind + " \n")
                    all_meta.append([pid, kind])
                    all_inputs.append(f"{question} \\n {candidates} \\n {paragraph}")

                get_question_and_candidates(line_split[3:8])
                get_question_and_candidates(line_split[8:13])
                get_question_and_candidates(line_split[13:18])
                get_question_and_candidates(line_split[18:23])

        # try:
        with open(file.replace(".tsv", ".ans")) as fans:
            for l in fans.readlines():
                all_answers.extend(l.replace("\n", "").split("\t"))
        # except (FileNotFoundError):
        #     pass

        assert len(all_answers) == len(all_inputs)

        for i, y in enumerate(all_answers):
            # fdataset_idx.write(all_inputs[i] + "\t" + y + "\n")
            correct_ans_idx = ord(y) - ord('A')
            fmeta.write(all_meta[i][0] + "\t" + y + "\t" + all_meta[i][1] + " \n")
            fdataset_string.write(all_inputs[i] + "\t" + all_candidates[i][correct_ans_idx] + "\n")

    read_and_convert_mctest_data('../datasets/mctest-master/data/MCTest/mc160.dev.tsv', "mc160", "dev")
    read_and_convert_mctest_data('../datasets/mctest-master/data/MCTest/mc160.train.tsv', "mc160", "train")

    read_and_convert_mctest_data('../datasets/mctest-master/data/MCTest/mc500.dev.tsv', "mc500", "dev")
    read_and_convert_mctest_data('../datasets/mctest-master/data/MCTest/mc500.train.tsv', "mc500", "train")

    # read_and_convert_mctest_data('../datasets/mctest-master/data/MCTest/mc160.dev.tsv', "mctest_corrected_the_separator", "dev", 'a')
    # read_and_convert_mctest_data('../datasets/mctest-master/data/MCTest/mc160.train.tsv', "mctest_corrected_the_separator", "train", 'a')
    # read_and_convert_mctest_data('../datasets/mctest-master/data/MCTest/mc500.dev.tsv', "mctest_corrected_the_separator", "dev", 'a')
    # read_and_convert_mctest_data('../datasets/mctest-master/data/MCTest/mc500.train.tsv', "mctest_corrected_the_separator", "train", 'a')


def read_and_parse_multiqa(file, dataset, kind):
    fout = open(f"{dataset}/{kind}.tsv", "w+")
    fmeta = open(f"{dataset}/{kind}_meta.txt", "w+")
    ans = open(f"{dataset}/{kind}_ans.jsonl", "w+")

    with open(file) as f:
        for l in f.readlines()[1:]:
            json_line = json.loads(l)
            pid = json_line['id']
            paragraph = ""
            for p in json_line['context']['documents']:
                if 'title' in p:
                    paragraph += f" ({p['title']}) "
                paragraph += p['text']
            paragraph = paragraph.strip().replace("\n", "").replace("\t", "")
            for q in json_line['qas']:
                qid = q['qid']
                fmeta.write(f"{pid}, {qid} \n")
                question = q['question']
                answers = []
                print(q)
                if 'cannot_answer' in q['answers']['open-ended']:
                    if q['answers']['open-ended']['cannot_answer'] == 'yes':
                        answers.append('<No Answer>')
                else:
                    for a in q['answers']['open-ended']['annotators_answer_candidates']:
                        print(a)
                        if 'extractive' in a['single_answer']:
                            answers.append(a['single_answer']['extractive']['answer'])
                        elif 'yesno' in a['single_answer']:
                            answers.append(a['single_answer']['yesno'])
                        else:
                            print("yo yo yo ")

                assert len(answers) > 0

                paragraph = paragraph.replace("\t", "").replace("   ", " ").replace("  ", " ").replace("\n", " ")
                question = question.replace("\t", "").replace("   ", " ").replace("  ", " ").replace("\n", " ")
                if '?' not in question:
                    question = question + "?"
                all_ans = [a.replace("\t", "").replace("   ", " ").replace("  ", " ").replace("\n", " ") for a in
                           answers]

                print(all_ans)
                fout.write(f"{question.strip()} \\n {paragraph.strip()}\t{all_ans[0].strip()}\n")
                ans.write(json.dumps(all_ans) + "\n")


def mkdir(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


from collections import Counter


def race(variation, grade, write_format="w+"):
    print(f">>>> race variation: {variation} / {grade}")
    assert variation == "idx" or variation == "string" or variation == 'string_no_candidates'

    count_map = {}

    def process_race_dir(kind):
        counter = {"counter": 0}
        if variation == "idx":
            dir = f"race_idx_{grade}"
            mkdir(dir)
            fin = open(f"{dir}/{kind.split('/')[0]}.tsv", write_format)
            fmeta = open(f"{dir}/{kind.split('/')[0]}_meta.txt", write_format)
        elif variation == "string":
            dir = f"race_string_{grade}"
            mkdir(dir)
            fin = open(f"{dir}/{kind.split('/')[0]}.tsv", write_format)
            fmeta = open(f"{dir}/{kind.split('/')[0]}_meta.txt", write_format)
        elif variation == "string_no_candidates":
            dir = f"race_string_no_candidates_{grade}"
            mkdir(dir)
            fin = open(f"{dir}/{kind.split('/')[0]}.tsv", write_format)
            fmeta = open(f"{dir}/{kind.split('/')[0]}_meta.txt", write_format)
        else:
            raise AttributeError

        def read_and_parse_race(file):
            with open(file) as f:
                counter["counter"] += 1
                line = f.readlines()[0]
                line = line.replace("\n", " ")
                jsonline = json.loads(line)
                answers = jsonline['answers']
                options = jsonline['options']
                questions = jsonline['questions']
                article = jsonline['article']
                article = article.replace("\n", " ").replace("\t", " ")
                id = jsonline['id']
                for i, q in enumerate(questions):
                    options[i] = [x.replace("\n", " ") for x in options[i]]
                    q = q.replace("\n", " ")
                    candidates = ("".join([f" ({chr(ord('A') + i)}) {x}" for i, x in enumerate(options[i])])).replace(
                        "\n", " ")
                    answer_idx = ord(answers[i]) - ord('A')
                    if variation == "idx":
                        fin.write(f"{q} \\n {candidates} \\n {article}\t{answers[i]} \n")
                    elif variation == "string":
                        fin.write(f"{q} \\n {candidates} \\n {article}\t{options[i][answer_idx]} \n")
                    elif variation == "string_no_candidates":
                        fin.write(f"{q} \\n {article} \t {options[i][answer_idx]} \n")
                    else:
                        raise AttributeError

                    fmeta.write(f"{id}\t{answers[i]}\n")

        directory_address = f"../datasets/RACE/{kind}/"
        directory = os.fsencode(directory_address)
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith(".txt"):
                read_and_parse_race(directory_address + filename)
            else:
                continue

        count_map[kind.split("/")[0]] = counter["counter"]

    process_race_dir(f"dev/{grade}")
    process_race_dir(f"test/{grade}")
    process_race_dir(f"train/{grade}")

    count_file = open(f"race_{variation}_{grade}/counts.json", "w+")
    count_file.write(json.dumps(count_map))


def newsqa():
    read_and_parse_multiqa("../datasets/NewsQA_dev.jsonl", "newsqa", "dev")
    read_and_parse_multiqa("../datasets/NewsQA_train.jsonl", "newsqa", "train")


def hotpotqa():
    read_and_parse_multiqa("../datasets/HotpotQA_dev.jsonl", "hotpotqa", "dev")
    read_and_parse_multiqa("../datasets/HotpotQA_train.jsonl", "hotpotqa", "train")


def squad():
    read_and_parse_multiqa("../datasets/SQuAD1-1_dev.jsonl", "squad1_1", "dev")
    read_and_parse_multiqa("../datasets/SQuAD1-1_train.jsonl", "squad1_1", "train")


def squad2():
    read_and_parse_multiqa("../datasets/SQuAD2-0_dev.jsonl", "squad2", "dev")
    read_and_parse_multiqa("../datasets/SQuAD2-0_train.jsonl", "squad2", "train")


def triviaqa():
    read_and_parse_multiqa("../datasets/TriviaQA_wiki_train.jsonl", "triviaqa", "train")
    read_and_parse_multiqa("../datasets/TriviaQA_wiki_dev.jsonl", "triviaqa", "dev")


def searchqa():
    read_and_parse_multiqa("../datasets/SearchQA_dev.jsonl", "searchqa", "dev")
    read_and_parse_multiqa("../datasets/SearchQA_train.jsonl", "searchqa", "train")


def boolq():
    read_and_parse_multiqa("../datasets/BoolQ_dev.jsonl", "boolq", "dev")
    read_and_parse_multiqa("../datasets/BoolQ_train.jsonl", "boolq", "train")


def duo_rc():
    read_and_parse_multiqa("../datasets/DuoRC_Paraphrase_dev.jsonl", "duo_rc_paragraph", "dev")
    read_and_parse_multiqa("../datasets/DuoRC_Paraphrase_train.jsonl", "duo_rc_paragraph", "train")

    read_and_parse_multiqa("../datasets/DuoRC_Self_dev.jsonl", "duo_rc_self", "dev")
    read_and_parse_multiqa("../datasets/DuoRC_Self_train.jsonl", "duo_rc_self", "train")


def drop():
    def load_file(name, dir):
        ftargets = open(f"{dir}/{name}_targets.txt", "+w")
        finput = open(f"{dir}/{name}_inputs.txt", "+w")
        fout = open(f"{dir}/{name}.tsv", "+w")
        fmeta = open(f"{dir}/{name}_meta.txt", "+w")
        span_lens = []
        with open(f"../datasets/drop_dataset/drop_dataset_{name}.json") as f:
            whole_data = json.load(f)
            for key in whole_data.keys():
                # print("------")
                # print(key)
                content = whole_data[key]
                passage = content['passage'].replace("\t", " ").replace("\n", " ")
                qa_pairs = content['qa_pairs']
                for qpair in qa_pairs:
                    ans_text = ""
                    question = qpair['question'].replace("\t", " ").replace("\n", " ")
                    answer = qpair['answer']
                    # print(answer)
                    number = answer['number']
                    spans = answer['spans']
                    if len(spans) > 0:
                        span_lens.append(len(spans))
                        ans_text = ", ".join(spans)
                    elif len(number) > 0:
                        ans_text = number
                    else:
                        day = answer['date']['day']
                        month = answer['date']['month']
                        year = answer['date']['year']
                        if len(month) > 0:
                            ans_text += month
                        if len(day) > 0:
                            ans_text += f" {day}"
                        if len(year) > 0:
                            ans_text += f" {year}"

                    # assert ans_text != ""
                    # print(ans_text)
                    if ans_text == "":
                        print(" >>>> skipping the question . . . ")
                        continue
                    ans_text = ans_text.replace("\t", " ").replace("\n", " ")
                    query_id = qpair['query_id']
                    fout.write(f"{question} \\n {passage}\t{ans_text}\n")
                    ftargets.write(f"{ans_text}\n")
                    finput.write(f"{question} \\n {passage}\n")
                    fmeta.write(f" {query_id}")

        print(span_lens)

    load_file("dev", "drop")
    load_file("train", "drop")


# def wikihop():
#     read_and_parse_multiqa("../datasets/BoolQ_dev.jsonl", "boolq", "dev")
#     read_and_parse_multiqa("../datasets/BoolQ_train.jsonl", "boolq", "train")
#
# def duorc_para():
#     read_and_parse_multiqa("../datasets/BoolQ_dev.jsonl", "boolq", "dev")
#     read_and_parse_multiqa("../datasets/BoolQ_train.jsonl", "boolq", "train")
#
# def duorc_self():
#     read_and_parse_multiqa("../datasets/BoolQ_dev.jsonl", "boolq", "dev")
#     read_and_parse_multiqa("../datasets/BoolQ_train.jsonl", "boolq", "train")
#
# def complex_questions():
#     read_and_parse_multiqa("../datasets/BoolQ_dev.jsonl", "boolq", "dev")
#     read_and_parse_multiqa("../datasets/BoolQ_train.jsonl", "boolq", "train")

# def comqa():
#     read_and_parse_multiqa("../datasets/BoolQ_dev.jsonl", "boolq", "dev")
#     read_and_parse_multiqa("../datasets/BoolQ_train.jsonl", "boolq", "train")


def extract_oyvind_predictions(file):
    all_predictions = {}
    with open(file) as f:
        for line in f.readlines():
            jsonline = json.loads(line)
            id = jsonline['id']
            if id not in all_predictions:
                all_predictions[id] = jsonline
            else:
                raise EnvironmentError
    return all_predictions


oyvind_test_preds = [
    [
        "roberta-combo",
        extract_oyvind_predictions("../datasets/oyvind_predictions/roberta-combo/eval_test.jsonl")
    ],
    [
        "roberta-no-ir",
        extract_oyvind_predictions("../datasets/oyvind_predictions/roberta-no-ir/eval_test.jsonl")
    ],
    [
        "roberta-question-stem-ir",
        extract_oyvind_predictions("../datasets/oyvind_predictions/roberta-question-stem-ir/eval_test.jsonl")
    ],
    [
        "roberta-standard-ir",
        extract_oyvind_predictions("../datasets/oyvind_predictions/roberta-standard-ir/eval_test.jsonl")
    ]
]

oyvind_dev_preds = [
    [
        "roberta-combo",
        extract_oyvind_predictions("../datasets/oyvind_predictions/roberta-combo/eval_validation.jsonl")
    ],
    [
        "roberta-no-ir",
        extract_oyvind_predictions("../datasets/oyvind_predictions/roberta-no-ir/eval_validation.jsonl")
    ],
    [
        "roberta-question-stem-ir",
        extract_oyvind_predictions("../datasets/oyvind_predictions/roberta-question-stem-ir/eval_validation.jsonl")
    ],
    [
        "roberta-standard-ir",
        extract_oyvind_predictions("../datasets/oyvind_predictions/roberta-standard-ir/eval_validation.jsonl")
    ]
]


def arc():
    directory_easy = "ARC-V1-Feb2018-2/ARC-Easy/ARC-Easy"
    directory_hard = "ARC-V1-Feb2018-2/ARC-Challenge/ARC-Challenge"

    def read_file(dir, split, kind, predictions_files, with_para=False):
        outdir = f"arc_{kind}"
        if with_para:
            outdir = f"arc_{kind}_with_ir"
        fout = open(f"{outdir}/{split.lower()}.tsv", "w+")
        fout_meta = open(f"{outdir}/{split.lower()}_meta.tsv", "w+")
        output_files = []
        if predictions_files:
            for x in predictions_files:
                fout_tmp = open(f"arc_{kind}/predictions_{x[0]}_{split}.txt", "w")
                output_files.append(fout_tmp)
                print(fout_tmp)

        correctness_map = {}
        with open(f"{dir}-{split}.jsonl") as f:
            for line in f.readlines():
                json_line = json.loads(line)
                question = json_line['question']['stem']
                choices = json_line['question']['choices']
                if kind == "easy":
                    id = "ARCEZ_" + json_line['id']
                else:
                    id = "ARCCH_" + json_line['id']
                para = ""
                if with_para:
                    print("done")
                    para = "\\n" + oyvind_paragraphs[id].replace("\n", " ").replace("\t", " ")
                # print(json_line)
                answer_key = json_line['answerKey']
                numbers = ""
                if 'A' in [c['label'] for c in choices]:
                    answer_key_idx = ord(answer_key[0]) - ord('A')
                    answer_label = answer_key[0]
                else:
                    answer_key_idx = ord(answer_key[0]) - ord('1')
                    answer_label = chr(ord(answer_key[0]) - ord('1') + ord('A'))
                    numbers = "numerical"

                candidates = " ".join([f"({chr(ord('A') + i)}) {c['text']}" for i, c in enumerate(choices)]).replace(
                    "\n", " ")

                # print((answer_key_idx, answer_key, candidates))
                answer_text = choices[answer_key_idx]['text']
                fout.write(f"{question} \\n {candidates}{para}\t{answer_text}\n")
                fout_meta.write(f"{json_line['id']}\t{answer_label}\t{numbers}\n")
                # fout_meta.write(f"{json_line['id']},{json_line['answerKey'][0]}\n")

                if predictions_files:
                    for i, x in enumerate(predictions_files):
                        pred_type = x[0]
                        predictions = x[1]
                        fout_tmp = output_files[i]
                        # print(f" ** pred type: {pred_type}")
                        if id not in predictions:
                            print(" >>>>> id not found . . . ")
                            # hack: use the gold ans
                            fout_tmp.write(answer_text + "\n")
                        else:
                            pred_json = predictions[id]

                            choice_text_list = pred_json['choice_text_list']
                            correct_answer_index = pred_json['correct_answer_index']
                            # label_probs = pred_json['label_probs']
                            answer_index = pred_json['answer_index']
                            fout_tmp.write(choice_text_list[answer_index] + "\n")

                            if pred_type not in correctness_map:
                                correctness_map[pred_type] = []
                            correctness_map[pred_type].append(1.0 if answer_index == correct_answer_index else 0.0)

        for pred_type in correctness_map.keys():
            if len(correctness_map[pred_type]) > 0:
                print(len(correctness_map[pred_type]))
                print(
                    f" **** Accuracy on {split} of ARC-{kind} ({pred_type}): {sum(correctness_map[pred_type]) / len(correctness_map[pred_type])}")

    for with_para in [True, False]:
        read_file(directory_easy, "Dev", "easy", oyvind_dev_preds, with_para)
        read_file(directory_easy, "Test", "easy", oyvind_test_preds, with_para)
        read_file(directory_easy, "Train", "easy", None, with_para)

        read_file(directory_hard, "Dev", "hard", oyvind_dev_preds, with_para)
        read_file(directory_hard, "Test", "hard", oyvind_test_preds, with_para)
        read_file(directory_hard, "Train", "hard", None, with_para)


def ai2_science():
    directory_middle = "../datasets/AI2-ScienceQuestions-V2.1-Jan2018/MiddleSchool/Middle-"
    directory_elementary = "../datasets/AI2-ScienceQuestions-V2.1-Jan2018/ElementarySchool/Elementary-"

    def read_file(dir, split, grade):
        fout = open(f"ai2_science_{grade.lower()}/{split}.tsv".lower(), "w+")
        foutmeta = open(f"ai2_science_{grade.lower()}/{split}_meta.tsv".lower(), "w+")
        with open(f"{dir}NDMC-{split.lower()}.jsonl") as f:
            for line in f.readlines():
                json_line = json.loads(line)
                question = json_line['question']['stem']
                choices = json_line['question']['choices']
                candidates = " ".join([f"({c['label']}) {c['text']}" for c in choices]).replace("\n", " ")
                print(json_line)
                answer_key = json_line['answerKey']
                answer_key_idx = ord(answer_key[0]) - ord('A')
                answer_text = choices[answer_key_idx]['text']
                fout.write(f"{question} \\n {candidates}\t{answer_text}\n")
                foutmeta.write(f"{json_line['id']}\t{answer_key[0]}\n")

    read_file(directory_middle, "Dev", "Middle")
    read_file(directory_middle, "Test", "Middle")
    read_file(directory_middle, "Train", "Middle")

    read_file(directory_elementary, "Dev", "Elementary")
    read_file(directory_elementary, "Test", "Elementary")
    read_file(directory_elementary, "Train", "Elementary")


def quoref():
    def read_file(file, segment):
        fout = open(f"quoref/{segment}.tsv", "w+")
        ftargets = open(f"quoref/{segment}_targets.txt", "+w")
        finputs = open(f"quoref/{segment}_inputs.txt", "+w")
        ans_size = []
        with open(file) as f:
            file = json.load(f)
            for section in file['data']:
                title = section['title'].replace("\n", " ").replace("\t", " ")
                for para in section['paragraphs']:
                    context = para['context'].replace("\n", " ").replace("\t", " ")
                    for qa in para['qas']:
                        question = qa['question'].replace("\n", " ").replace("\t", " ")
                        ans_size.append(len(qa['answers']))
                        for a in qa['answers']:
                            answer = a['text'].replace("\n", " ").replace("\t", " ")
                            fout.write(f"{question} \\n ({title}) {context}\t{answer}\n")
                            ftargets.write(f"{answer}\n")
                            finputs.write(f"{question} \\n ({title}) {context}\n")
        print(sum(ans_size) / len(ans_size))

    read_file("../datasets/quoref-train-dev-v0.1/quoref-dev-v0.1.json", "dev")
    read_file("../datasets/quoref-train-dev-v0.1/quoref-train-v0.1.json", "train")


def ropes():
    def read_file(file, segment):
        ans_size = []
        fout = open(f"ropes/{segment}.tsv", "w+")
        ftargets = open(f"ropes/{segment}_targets.txt", "+w")
        finput = open(f"ropes/{segment}_inputs.txt", "+w")
        with open(file) as f:
            file = json.load(f)
            for section in file['data']:
                for para in section['paragraphs']:
                    context = f"{para['background']} {para['situation']}".replace("\n", " ").replace("\t", " ")
                    for qa in para['qas']:
                        question = qa['question'].replace("\n", " ").replace("\t", " ")
                        ans_size.append(len(qa['answers']))
                        for a in qa['answers']:
                            answer = a['text'].replace("\n", " ").replace("\t", " ")
                            fout.write(f"{question} \\n {context}\t{answer}\n")
                            ftargets.write(f"{answer}\n")
                            finput.write(f"{question} \\n {context}\n")

    read_file("../datasets/ropes-train-dev-v1.0/dev-v1.0.json", "dev")
    read_file("../datasets/ropes-train-dev-v1.0/train-v1.0.json", "train")


def narrative_qa():
    paragraphs = {}
    with open("../datasets/narrativeqa/third_party/wikipedia/summaries.csv") as f:
        spamreader = csv.reader(f)
        for i, line in enumerate(spamreader):
            print(line)
            if i == 0:
                continue
            paragraphs[line[0]] = line[2].replace("\n", "")

    fout_test = open(f"narrativeqa/test.tsv", "w+")
    fout_train = open(f"narrativeqa/train.tsv", "w+")
    fout_dev = open(f"narrativeqa/dev.tsv", "w+")
    counts = open(f"narrativeqa/counts.json", "w+")

    count_train = 0
    count_test = 0
    count_dev = 0
    with open("..//datasets/narrativeqa/qaps.csv") as f:
        spamreader = csv.reader(f)
        for i, line in enumerate(spamreader):
            print(line)
            if i == 0:
                continue
            line1 = f"{line[2]} \\n {paragraphs[line[0]]} \t {line[3]} \n"
            line2 = f"{line[2]} \\n {paragraphs[line[0]]} \t {line[4]} \n"
            if line[1] == "train":
                fout_train.write(line1)
                fout_train.write(line2)
                count_train += 1
            elif line[1] == "test":
                fout_test.write(line1)
                fout_test.write(line2)
                count_test += 1
            elif line[1] == "valid":
                fout_dev.write(line1)
                fout_dev.write(line2)
                count_dev += 1
            else:
                print(" >>>> ERROR ")

    counts.write(json.dumps({"train": count_train, "dev": count_dev, "test": count_test}))


def multirc():
    def read_file(file):
        lines = []
        with open(f"../datasets/multirc/{file}") as f:
            for line in f.readlines():
                line_split = line.split("\t")
                paragraph = line_split[4].replace("\n", " ").replace("\t", " ")
                question = line_split[5].replace("\n", " ").replace("\t", " ")
                line_split[6] = line_split[6].replace("\n", "")
                assert line_split[6] == "True" or line_split[6] == "False", f"`{line_split[6]}`"
                answer = "yes" if line_split[6] == "True" else "no"
                lines.append(f"{question} \\n {paragraph}\t{answer}\n")
        return lines

    lines1 = read_file("dev_83-fixedIds.json.yes-nos.tsv")
    lines2 = read_file("train_456-fixedIds.json.yes-nos.tsv")
    fout = open(f"multirc/dev.tsv", "w+")
    fout2 = open(f"multirc/train.tsv", "w+")
    for line in lines1 + lines2:
        fout.write(line)
        fout2.write(line)


def openbookqa():
    def read_file(file, split, predictions_files, with_para=False):
        out_dir = "openbookqa"
        if with_para:
            out_dir = "openbookqa_with_ir"
        fout = open(f"{out_dir}/{split}.tsv", "w+")
        fout_meta = open(f"{out_dir}/{split}_meta.tsv", "w+")
        output_files = []
        oyind_accuracy = {}
        if predictions_files:
            fout_target_tmp = open(f"openbookqa/oyvind/_target.txt", "w")
            for x in predictions_files:
                fout_tmp = open(f"openbookqa/oyvind/predictions_{x[0]}_{split}.txt", "w")
                output_files.append(fout_tmp)
                # print(fout_tmp)
                oyind_accuracy[x] = []

        with open(file) as f:
            for line in f.readlines():
                json_line = json.loads(line)
                question = json_line['question']['stem']
                choices = json_line['question']['choices']
                candidates = " ".join([f"({c['label']}) {c['text']}" for c in choices]).replace("\n", " ")
                print(json_line)
                answer_key = json_line['answerKey']
                answer_key_idx = ord(answer_key[0]) - ord('A')
                answer_text = choices[answer_key_idx]['text']
                id = "OBQA_" + json_line['id']
                para = ""
                if with_para:
                    para = "\\n" + oyvind_paragraphs[id].replace("\n", " ").replace("\t", " ")
                fout.write(f"{question} \\n {candidates}{para}\t{answer_text}\n")
                fout_meta.write(f"{json_line['id']}\t{answer_key[0]}\n")
                if predictions_files:
                    fout_target_tmp.write(f"{answer_text}\n")
                    for i, x in enumerate(predictions_files):
                        pred_type = x[0]
                        predictions = x[1]
                        fout_tmp = output_files[i]
                        # print(f" ** pred type: {pred_type}")
                        if id not in predictions:
                            print(" >>>>> id not found . . . ")
                            # hack: use the gold ans
                            fout_tmp.write(answer_text + "\n")
                        else:
                            pred_json = predictions[id]

                            choice_text_list = pred_json['choice_text_list']
                            # correct_answer_index = pred_json['correct_answer_index']
                            answer_index = pred_json['answer_index']
                            fout_tmp.write(choice_text_list[answer_index] + "\n")
                            if answer_index == answer_key_idx:
                                oyind_accuracy[x].append(1.0)
                            else:
                                oyind_accuracy[x].append(0.0)

            if predictions_files:
                for x in predictions_files:
                    print(f" *** {x} \t accuracy: {sum(predictions_files[x]) / len(predictions_files)} ")

    for with_para in [True, False]:
        read_file("../datasets/OpenBookQA-V1-Sep2018/Data/Main/dev.jsonl", "dev", None, with_para)
        # read_file("../datasets/OpenBookQA-V1-Sep2018/Data/Main/test.jsonl", "test", oyvind_test_preds, with_para)
        read_file("../datasets/OpenBookQA-V1-Sep2018/Data/Main/test.jsonl", "test", None, with_para)
        read_file("../datasets/OpenBookQA-V1-Sep2018/Data/Main/train.jsonl", "train", None, with_para)


def boolq_np():
    outfile = {
        "dev": open("/Users/danielk/ideaProjects/t2t-qa/t2t-data/boolq-np/dev.tsv", "w"),
        "train": open("/Users/danielk/ideaProjects/t2t-qa/t2t-data/boolq-np/train.tsv", "w"),
    }
    with open("boolq_natural_perturbations.jsonl") as f:
        for line in f.readlines():
            json_line = json.loads(line)
            # print(json_line['split'])
            if json_line['is_seed_question'] == 1:
                json_line['question'] += '?'
            label = "yes" if json_line['hard_label'] == "True" else "no"
            outfile[json_line['split']].write(f"{json_line['question']}\\n{json_line['passage']}\t{label}\n")


def read_paragraphs(file):
    map = {}
    with open(file) as f:
        for line in f.readlines():
            json_line = json.loads(line)
            map[json_line['id']] = json_line['para']

    return map


oyvind_paragraphs = {}
map1 = read_paragraphs("oyvind_arc_obqa_reg_with_ir/train.jsonl")
map2 = read_paragraphs("oyvind_arc_obqa_reg_with_ir/test.jsonl")
map3 = read_paragraphs("oyvind_arc_obqa_reg_with_ir/dev.jsonl")
oyvind_paragraphs.update(map1)
oyvind_paragraphs.update(map2)
oyvind_paragraphs.update(map3)


def ambigqa():
    def read_file(file, dir, split):
        outfile = open(f"{dir}/{split}.tsv", "+w")
        outfile_meta = open(f"{dir}/{split}_meta.tsv", "+w")
        size = 0
        with open(file, "r") as f:
            json_file = json.load(f)
            for item in tqdm(json_file):
                question = item['question'].replace("\n", " ").replace("\t", " ")
                single_answers_already_included = []
                for anno in item["annotations"]:
                    if anno['type'] == "singleAnswer":
                        for ans in anno['answer']:
                            if ans not in single_answers_already_included:
                                ans = ans.replace("\n", " ").replace("\t", " ")
                                outfile.write(f"{question}\t{ans}\n")
                                outfile_meta.write(item['id'] + "\n")
                                single_answers_already_included.append(ans)
                                size += 1
                    else:
                        answers = []
                        for x in anno['qaPairs']:
                            answers.append(x['answer'][0])

                        answers = [x.strip() for x in answers]
                        answers = list(set(answers))  # to drop duplicate answers
                        for i, ordering in enumerate(itertools.permutations(answers)):
                            if i >= 3:
                                break
                            ans_str = " [SEP] ".join(ordering).replace("\n", " ").replace("\t", " ")
                            outfile.write(f"{question}\t{ans_str}\n")
                            outfile_meta.write(item['id'] + "\n")
                            size += 1
        return size

    count_dev = read_file("ambignq_light/dev_light.json", "ambigqa", "dev")
    count_train = read_file("ambignq_light/train_light.json", "ambigqa",
                            "train")
    count_test = 0

    # Create TSVs and get counts.
    with open("ambigqa/counts.json", "w") as outfile:
        json.dump({"train": count_train, "dev": count_dev, "test": count_test}, outfile)


def natural_questions_direct_answer():

    question_to_para_map = read_natural_questions_paragraphs()

    def read_file(in_fname, dir, split, with_paragraphs=False, aggregared_ans=False):
        outfile = open(f"{dir}/{split}.tsv", "+w")
        outfile_meta = open(f"{dir}/{split}_meta.tsv", "+w")
        with open(in_fname) as f:
            json_file = json.load(f)
            size = 0
            for i, item in enumerate(json_file):
                id = item['id']
                question = item['question'].replace("\t", " ").replace("\n", " ")
                if "?" not in question:
                    question += "?"
                para = ""
                if with_paragraphs:
                    para = question_to_para_map[f"{split}-{i}"].replace("\t", " ").replace("\n", " ").replace("[SEP]", "-").replace("[sep]", "-")
                    para = " ".join(para.split(" ")[1:600]) # take the subset
                    para = "\\n" + para
                if aggregared_ans:
                    answers = [answer.replace("\t", " ").replace("\n", " ") for answer in item['answer']]
                    random.shuffle(answers)
                    concatenated_answers = "///".join(answers)
                    outfile.write(f"{question}{para}\t{concatenated_answers}\t{answers[0]}\n")
                    outfile_meta.write(f"{id}\n")
                    size += 1
                else:
                    for answer in item['answer']:
                        answer = answer.replace("\t", " ").replace("\n", " ")
                        outfile.write(f"{question}{para}\t{answer}\n")
                        outfile_meta.write(f"{id}\n")
                        size += 1
        return size

    print("Generating NQ TSVs.")
    # Create TSVs and get counts.
    for dir in ['natural_questions_direct_ans_aggregated']: # ['natural_questions_direct_ans', 'natural_questions_with_dpr_para']:
        with_para = True if "dpr" in dir else False
        aggregared_ans = True if "aggregated" in dir else False
        count_dev = read_file("../datasets/nq/nqopen/nqopen-dev.json", dir, "dev", with_para, aggregared_ans)
        count_train = read_file("../datasets/nq/nqopen/nqopen-train.json", dir, "train", with_para, aggregared_ans)
        with open(dir + "/counts.json", "w") as outfile:
            json.dump({"train": count_train, "dev": count_dev}, outfile)

        count_train = read_file("../datasets/nq/nqopen/nqopen-train.json", dir + "_test", "train", with_para, aggregared_ans)
        count_test = read_file("../datasets/nq/nqopen/nqopen-test.json", dir + "_test", "test", with_para, aggregared_ans)
        with open(dir + "_test" + "/counts.json", "w") as outfile:
            json.dump({"train": count_train, "test": count_test}, outfile)

## NQ contexts
def read_natural_questions_paragraphs():
    question_to_para_map = {}
    def read_file(file, split):
        with open(file) as f:
            json_file = json.load(f)
            for i, item in enumerate(json_file):
                question_to_para_map[f"{split}-{i}"] = item['context']

    read_file("../datasets/nq-dpr-output/train.json", "train")
    read_file("../datasets/nq-dpr-output/test.json", "test")
    read_file("../datasets/nq-dpr-output/dev.json", "dev")

    return question_to_para_map


def natural_questions_reading_comprehension():
    def read_file(in_fname, out_fname):

        def extract_answer(tokens, span):
            """Reconstruct answer from token span and remove extra spaces."""
            start, end = span["start_token"], span["end_token"]
            ans = " ".join(tokens[start:end])
            # Remove incorrect spacing around punctuation.
            ans = ans.replace(" ,", ",").replace(" .", ".").replace(" %", "%")
            ans = ans.replace(" - ", "-").replace(" : ", ":").replace(" / ", "/")
            ans = ans.replace("( ", "(").replace(" )", ")")
            ans = ans.replace("`` ", "\"").replace(" ''", "\"")
            ans = ans.replace(" 's", "'s").replace("s ' ", "s' ")
            return ans

        count = 0
        with open(in_fname, "r") as infile, open(out_fname, "w") as outfile:
            for line in infile.readlines():
                ex = json.loads(line)
                # Remove any examples with more than one answer.
                if len(ex['annotations'][0]['short_answers']) != 1:
                    continue
                # Questions in NQ do not include a question mark.
                question = ex["question_text"] + "?"
                answer_span = ex['annotations'][0]['short_answers'][0]
                # Handle the two document formats in NQ (tokens or text).
                if "document_tokens" in ex:
                    tokens = [t["token"] for t in ex["document_tokens"]]
                elif "document_text" in ex:
                    tokens = ex["document_text"].split(" ")
                answer = extract_answer(tokens, answer_span)
                # Write this line as <question>\t<answer>
                outfile.write("%s\t%s\n" % (question, answer))
                count += 1
                if count % 1000 == 1:
                    print(f"Wrote {count} examples to {out_fname}.")
            return count

    count_dev = read_file("../datasets/dev-all.jsonl", "natural_questions/dev.tsv")
    count_train = read_file("../datasets/nq-train.jsonl", "natural_questions/train.tsv")

    # Create TSVs and get counts.
    print("Generating NQ TSVs.")
    with open("natural_questions/counts.json", "w") as outfile:
        json.dump({"train": count_train, "dev": count_dev}, outfile)


def winogrande():
    def read_file(size, split, outfolder):
        counter = 0
        outfile = open(f"{outfolder}/{split}.tsv", "w+")
        outfile_meta = open(f"{outfolder}/{split}_meta.tsv", "w+")
        file_name = f"{split}_{size}.jsonl"
        # label_file_name = f"{split}_{size}-labels.lst"
        if split != "train":
            file_name = f"{split}.jsonl"
            # label_file_name = f"{split}-labels.lst"

        with open(f"winogrande_1.1/{file_name}") as f:
            for line in f.readlines():
                json_line = json.loads(line)
                qID = json_line['qID']
                sentence = json_line['sentence']
                option1 = json_line['option1']
                option2 = json_line['option2']
                ans = ""
                idx = "-"
                idx_string = "-"
                if 'answer' in json_line:
                    idx = json_line['answer']
                    ans = option1
                    assert idx == "1" or idx == "2"
                    if idx == "2":
                        ans = option2
                        idx_string = "B"
                    else:
                        idx_string = "A"
                outfile.write(f"{sentence} \\n (A) {option1} (B) {option2} \t {ans} \n")
                outfile_meta.write(f"{qID}\t{idx_string}\t numeric \t {ans} \n")

                counter += 1
        return counter

    for size in ["xs", "s", "m", "l", "xl"]:
        train_count = read_file(size, "train", f"winogrande_{size}")
        dev_count = read_file(size, "dev", f"winogrande_{size}")
        # test_count = read_file(size, "test")

        with open(f"winogrande_{size}/counts.json", "w+") as outfile:
            json.dump({"train": train_count, "dev": dev_count}, outfile)

    train_count = read_file("s", "train", f"winogrande_test")
    test_count = read_file("s", "test", f"winogrande_test")
    with open(f"winogrande_test/counts.json", "w+") as outfile:
        json.dump({"train": train_count, "test": test_count}, outfile)

def anlg():
    director = "/Users/danielk/ideaProjects/t2t-qa/t2t-data/anlg_dev/"
    def readfile(inputfile, labelfile, split):
        labels = []
        with open(labelfile) as f1:
            for line in f1.readlines():
                labels.append(int(line.replace("\n", "")))
        outfile = open(director + split + ".tsv", "+w")
        outmetafile = open(director + split + "_meta.tsv", "+w")
        with open(inputfile) as f2:
            for idx, line in enumerate(f2.readlines()):
                label = labels[idx]
                assert label == 1 or label == 2, f" * the label is: {label}"
                json_line = json.loads(line)
                outstring = json_line['hyp1']
                if label == 2:
                    outstring = json_line['hyp2']
                outfile.write(json_line['obs1'] + " ___ "  + json_line['obs2'] + "\t" + outstring + "\n")
                outmetafile.write(f"{json_line['story_id']}\t{label}\n")

        return len(labels)

    dev_count = readfile("../datasets/aNLG/dev.jsonl", "../datasets/aNLG/dev-labels.lst", "dev")
    train_count = readfile("../datasets/aNLG/train.jsonl", "../datasets/aNLG/train-labels.lst", "train")

    with open(director + "counts.json", "w+") as outfile:
        json.dump({"train": train_count, "dev": dev_count}, outfile)

csv.field_size_limit(10 * 131072)

def summarization():
    def readfile(file):
        outfile = open(file.replace(".tsv", "_2.tsv"), "+w")
        with open(file) as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            for row in reader:
                row[0] = row[0].replace("\t", " ").replace("\n", " ")
                row[0] = row[0][:6*500]
                row[1] = row[1].replace("\t", " ").replace("\n", " ")
                outfile.write(row[0] + "\t" + row[1] + "\n")

    readfile("/Users/danielk/ideaProjects/t2t-qa/t2t-data/summarization-cnndm-dev/dev.tsv")
    readfile("/Users/danielk/ideaProjects/t2t-qa/t2t-data/summarization-cnndm-dev/train.tsv")

    readfile("/Users/danielk/ideaProjects/t2t-qa/t2t-data/summarization-cnndm-test/test.tsv")
    readfile("/Users/danielk/ideaProjects/t2t-qa/t2t-data/summarization-cnndm-test/train.tsv")

    readfile("/Users/danielk/ideaProjects/t2t-qa/t2t-data/summarization-xsum-dev/dev.tsv")
    readfile("/Users/danielk/ideaProjects/t2t-qa/t2t-data/summarization-xsum-dev/train.tsv")

    readfile("/Users/danielk/ideaProjects/t2t-qa/t2t-data/summarization-xsum-test/test.tsv")
    readfile("/Users/danielk/ideaProjects/t2t-qa/t2t-data/summarization-xsum-test/train.tsv")
    
    
def csqa2_process(file, dataset, kind):
    fout = open(f"{dataset}/{kind}.tsv", "w+")
    fmeta = open(f"{dataset}/{kind}_meta.txt", "w+")
    ans = open(f"{dataset}/{kind}_ans.jsonl", "w+")

    df=pd.read_json('/content/csqa2/dataset/'+file, lines=True, compression='gzip')
    questions=df[['question','answer','id']].values

    for row in range(len(questions)):
        question=questions[row][0].strip().replace("\n", "").replace("\t", "").replace("   ", " ").replace("  ", " ")
        if '?' not in question:
            question = question + "?"
        answer=[questions[row][1].strip().replace("\n", "").replace("\t", "").replace("   ", " ").replace("  ", " ")]
        id=questions[row][2]

        fmeta.write(f"{id} \n")
        fout.write(f"{question} \t{answer[0]}\n")
        ans.write(json.dumps(answer) + "\n")
    return len(questions)

def csqa2_process_test(file, dataset, kind):
    fout = open(f"{dataset}/{kind}.tsv", "w+")
    fmeta = open(f"{dataset}/{kind}_meta.txt", "w+")

    df=pd.read_json('/content/'+file, lines=True, compression='gzip')
    questions=df[['question','id']].values

    for row in range(len(questions)):
        question=questions[row][0].strip().replace("\n", "").replace("\t", "").replace("   ", " ").replace("  ", " ")
        if '?' not in question:
            question = question + "?"
        id=questions[row][1]
        answer=["-"]

        fmeta.write(f"{id} \n")
        fout.write(f"{question} \t{answer[0]}\n")    
    return len(questions)
    
def csqa():
    train_count = csqa2_process('CSQA2_train.json.gz','csqa2','train')
    dev_count = csqa2_process('CSQA2_dev.json.gz','csqa2','dev')
    test_count = csqa2_process_test('CSQA2_test_no_answers.json.gz','csqa2','test')
    with open(f"/content/csqa2/counts.json", "w+") as outfile:
        json.dump({"train": train_count, "dev": dev_count, "test": test_count}, outfile)
    
def pubmedqa_process(file, dataset, kind):
    fout_long = open(f"{dataset}/long_answer/{kind}.tsv", "w+")
    fout_short = open(f"{dataset}/short_answer/{kind}.tsv", "w+")
    fmeta = open(f"{dataset}/{kind}_meta.txt", "w+")
    ans_long = open(f"{dataset}/long_answer/{kind}_ans.jsonl", "w+")
    ans_short = open(f"{dataset}/short_answer/{kind}_ans.jsonl", "w+")

    df=pd.read_json(codecs.open('/content/'+file,'r','utf-8')).transpose()
    questions=df[['QUESTION','CONTEXTS','LONG_ANSWER','final_decision']].values
    meta=df.index.values
    for id in meta:
        fmeta.write(f"{id} \n")

    for row in range(len(questions)):
        question=questions[row][0].strip().replace("\n", "").replace("\t", "").replace("   ", " ").replace("  ", " ")
        if '?' not in question:
            question = question + "?"
        separator=','
        contexts=separator.join(questions[row][1]).strip().replace("\n", "").replace("\t", "").replace("   ", " ").replace("  ", " ")
        long_answer=[questions[row][2].strip().replace("\n", "").replace("\t", "").replace("   ", " ").replace("  ", " ")]
        answer=[questions[row][3].strip().replace("\n", "").replace("\t", "").replace("   ", " ").replace("  ", " ")]

        fout_long.write(f"{question}\\n {contexts} \t{long_answer[0]}\n")
        fout_short.write(f"{question}\\n {contexts} \t{answer[0]}\n")
        ans_short.write(json.dumps(answer) + "\n")
        ans_long.write(json.dumps(long_answer) + "\n")
    
def pubmedqa_process_un(file, dataset, kind):
    fout_long = open(f"{dataset}/long_answer/{kind}.tsv", "w+")
    fmeta = open(f"{dataset}/{kind}_meta.txt", "w+")
    ans_long = open(f"{dataset}/long_answer/{kind}_ans.jsonl", "w+")

    df=pd.read_json(codecs.open('/content/'+file,'r','utf-8')).transpose()
    questions=df[['QUESTION','CONTEXTS','LONG_ANSWER']].values
    meta=df.index.values
    for id in meta:
        fmeta.write(f"{id} \n")
        
    for row in range(len(questions)):
        question=questions[row][0].strip().replace("\n", "").replace("\t", "").replace("   ", " ").replace("  ", " ")
        if '?' not in question:
            question = question + "?"
        separator=','
        contexts=separator.join(questions[row][1]).strip().replace("\n", "").replace("\t", "").replace("   ", " ").replace("  ", " ")
        long_answer=[questions[row][2].strip().replace("\n", "").replace("\t", "").replace("   ", " ").replace("  ", " ")]

        fout_long.write(f"{question}\\n {contexts} \t{long_answer[0]}\n")
        ans_long.write(json.dumps(long_answer) + "\n")
    
def pubmedqa():
    pubmedqa_process('ori_pqal.json','pubmedqa','pqal_train')
    pubmedqa_process('ori_pqaa.json','pubmedqa','pqaa_train')
    pubmedqa_process('test_set.json','pubmedqa','test')
    pubmedqa_process_un('ori_pqau.json','pubmedqa','pqau_train')
    
def strategyqa_process(file, dataset, kind):
    fout = open(f"{dataset}/{kind}.tsv", "w+")
    fmeta = open(f"{dataset}/{kind}_meta.txt", "w+")
    ans = open(f"{dataset}/{kind}_ans.jsonl", "w+")

    df=pd.read_json(codecs.open('/content/'+file,'r','utf-8'))
    questions=df[['qid','term','question','answer']].values

    documents=pd.read_json(codecs.open('/content/queries_cache.json','r','utf-8'))

    for row in range(len(questions)):
        qid = questions[row][0]
        term = questions[row][1].strip().replace("\n", "").replace("\t", "").replace("   ", " ").replace("  ", " ")
        term = "(" +term + ")"
        question=questions[row][2].strip().replace("\n", "").replace("\t", "").replace("   ", " ").replace("  ", " ")
        if '?' not in question:
            question = question + "?"
        if questions[row][3]==True:
            answer=["yes"]
        else:
            answer=["no"]
        query=clean_query(questions[row][2]) 
        arr=documents[query]
        retrieved_documents=[]
        token_num=0
        for result in arr[0]:
            sentences=result["sentence"].split(".")
            for index in range(len(sentences)-1):
                if (token_num+len(sentences[index].split(" ")))<500:
                    token_num += len(sentences[index].split(" "))
                    retrieved_documents.append(sentences[index] + ".")
        retrieved_document=''.join(retrieved_documents).strip().replace("\n", "").replace("\t", "").replace("   ", " ").replace("  ", " ")
    
        fout.write(f"{question}\\n {term} {retrieved_document} \t{answer[0]}\n")
        ans.write(json.dumps(answer) + "\n")
        fmeta.write(f"{qid} \n")
    return len(questions)

def strategyqa_process_test(file, dataset, kind):
    fout = open(f"{dataset}/{kind}.tsv", "w+")
    fmeta = open(f"{dataset}/{kind}_meta.txt", "w+")
    
    df=pd.read_json(codecs.open('/content/'+file,'r','utf-8'))
    questions=df[['qid','question']].values

    documents=pd.read_json(codecs.open('/content/queries_cache.json','r','utf-8'))

    for row in range(len(questions)):
        qid = questions[row][0]
        question=questions[row][1].strip().replace("\n", "").replace("\t", "").replace("   ", " ").replace("  ", " ")
        if '?' not in question:
            question = question + "?"

        query=clean_query(questions[row][1]) 
        arr=documents[query]
        retrieved_documents=[]
        token_num=0
        answer=["-"]
        for result in arr[0]:
            sentences=result["sentence"].split(".")
            for index in range(len(sentences)-1):
                if (token_num+len(sentences[index].split(" ")))<500:
                    token_num += len(sentences[index].split(" "))
                    retrieved_documents.append(sentences[index] + ".")
        retrieved_document=''.join(retrieved_documents).strip().replace("\n", "").replace("\t", "").replace("   ", " ").replace("  ", " ")
    
        fout.write(f"{question}\\n {retrieved_document} \t{answer[0]}\n")
        fmeta.write(f"{qid} \n")
    return len(questions)

def clean_query(query, remove_stopwords=True):
    if remove_stopwords:
        query_split = query.split()
        new_query_split = []
        for word in query_split:
            if word.lower() + " " not in STOPWORDS:
                new_query_split.append(word)
        query = " ".join(new_query_split)
    return query

def strategyqa():
    train_count=strategyqa_process('strategyqa_train.json','strategyqa','train')
    test_count=strategyqa_process_test('strategyqa_test.json','strategyqa','test')
    with open(f"/content/strategyqa/counts.json", "w+") as outfile:
        json.dump({"train": train_count, "test": test_count}, outfile)
        
def reclor_process(file, dataset, kind):
    fout = open(f"{dataset}/{kind}.tsv", "w+")
    fmeta = open(f"{dataset}/{kind}_meta.txt", "w+")

    df=pd.read_json(codecs.open('/content/'+file,'r','utf-8'))
    questions=df[['question','answers','context','label','id_string']].values
        
    for row in range(len(questions)):
        question=questions[row][0].strip().replace("\n", "").replace("\t", "").replace("   ", " ").replace("  ", " ")
        if '?' not in question:
            question = question + "?"
        candidates=questions[row][1]        
        options = " ".join([f"({chr(ord('A') + i)}) {x}" for i, x in enumerate(candidates)])
        contexts=questions[row][2].strip().replace("\n", "").replace("\t", "").replace("   ", " ").replace("  ", " ")

        label = questions[row][3]
        answer=questions[row][1][label].strip().replace("\n", "").replace("\t", "").replace("   ", " ").replace("  ", " ")
        id=questions[row][4]
        answer_index = chr(ord('A')+(label))

        fmeta.write(f"{id}\t{answer_index}\n")
        fout.write(f"{question} \\n{options} \\n {contexts}\t{answer}\n")
    return len(questions)

def reclor_process_test(file, dataset, kind):
    fout = open(f"{dataset}/{kind}.tsv", "w+")
    fmeta = open(f"{dataset}/{kind}_meta.txt", "w+")

    df=pd.read_json(codecs.open('/content/'+file,'r','utf-8'))
    questions=df[['question','answers','context','id_string']].values
        
    for row in range(len(questions)):
        question=questions[row][0].strip().replace("\n", "").replace("\t", "").replace("   ", " ").replace("  ", " ")
        if '?' not in question:
            question = question + "?"
        candidates=questions[row][1]        
        options = " ".join([f"({chr(ord('A') + i)}) {x}" for i, x in enumerate(candidates)])
        contexts=questions[row][2].strip().replace("\n", "").replace("\t", "").replace("   ", " ").replace("  ", " ")
        answer_index="-"
        answer="-"
        id=questions[row][3]

        fmeta.write(f"{id}\t{answer_index}\n")
        fout.write(f"{question} \\n{options} \\n {contexts}\t{answer}\n")
    return len(questions)

def reclor():
    train_count = reclor_process("train.json","reclor","train")
    val_count = reclor_process("val.json","reclor","val")
    test_count = reclor_process_test("test.json","reclor","test")
    with open(f"/content/reclor/counts.json", "w+") as outfile:
        json.dump({"train": train_count, "val": val_count, "test": test_count}, outfile)

def race_c_process(dataset, kind):
    fout = open(f"{dataset}/{kind}.tsv", "w+")
    fmeta = open(f"{dataset}/{kind}_meta.txt", "w+")

    l = [pd.read_json(filename) for filename in glob.glob("/content/data/"+kind+"/*.txt")]
    df = pd.concat(l, axis=0)
    questions=df[['questions','options','article','answers','id']].values
        
    for row in range(len(questions)):
        question=questions[row][0].strip().replace("\n", "").replace("\t", "").replace("   ", " ").replace("  ", " ")
        if '?' not in question:
            question = question + "?"
        candidates=questions[row][1]        
        options = " ".join([f"({chr(ord('A') + i)}) {x}" for i, x in enumerate(candidates)])
        contexts=questions[row][2].strip().replace("\n", "").replace("\t", "").replace("   ", " ").replace("  ", " ")

        answer = questions[row][3]
        answer_index=ord(answer)-ord('A')
        answer_string=questions[row][1][answer_index].strip().replace("\n", "").replace("\t", "").replace("   ", " ").replace("  ", " ")
        id=questions[row][4]

        fmeta.write(f"{id}\t{answer} \n")
        fout.write(f"{question} \\n{options} \\n {contexts}\t{answer_string} \n")
    return len(questions)

def race_c():
    train_count= race_c_process("race-c","train")
    dev_count= race_c_process("race-c","dev")
    test_count= race_c_process("race-c","test")
    with open(f"/content/race-c/counts.json", "w+") as outfile:
        json.dump({"train": train_count, "dev": dev_count, "test": test_count}, outfile)
        
def record_process_extractive (file,dataset, kind):
    fout = open(f"{dataset}/{kind}.tsv", "w+")
    fmeta = open(f"{dataset}/{kind}_meta.txt", "w+")
    counter=0
    
    with open("/content/"+file) as f:
        for l in f.readlines()[1:]:
            json_line = json.loads(l)
            contexts=json_line['passage']['text'].replace("@highlight","")
            contexts=contexts.strip().replace("\n", "").replace("\t", "").replace("   ", " ").replace("  ", " ")
            for index in range(len(json_line['qas'])):
                counter +=1
                question=json_line['qas'][index]['query'].replace("@placeholder","_")
                question=question.replace("\n", "").replace("\t", "").replace("   ", " ").replace("  ", " ")
                if '.' not in question:
                    question = question + "."  
                if kind is not "test":     
                    answer_string=json_line['qas'][index]['answers'][0]['text'].strip().replace("\n", "").replace("\t", "").replace("   ", " ").replace("  ", " ")
                else:
                    answer_string="-"
                id=json_line['qas'][index]['idx']

                fmeta.write(f"{id}\n")
                fout.write(f"{question} \\n {contexts} \t {answer_string} \n")
    return counter

def record_extractive():
    train_count=record_process_extractive("train.jsonl","record","train")
    val_count=record_process_extractive("val.jsonl","record","val")
    test_count=record_process_extractive("test.jsonl","record","test")
    with open(f"/content/record/counts.json", "w+") as outfile:
        json.dump({"train": train_count, "val": val_count, "test": test_count}, outfile)
        
def record_process_mc(file,dataset, kind):
    fout = open(f"{dataset}/{kind}.tsv", "w+")
    fmeta = open(f"{dataset}/{kind}_meta.txt", "w+")
    counter=0
    
    with open("/content/"+file) as f:
        for l in f.readlines()[1:]:
            json_line = json.loads(l)
            for index in range(len(json_line['qas'])):
                counter +=1
                contexts=json_line['passage']['text']
                entities=json_line['passage']['entities']
                options=[]
                answers=[]
                if kind is not "test":
                    for row in range(len(json_line['qas'][index]['answers'])):
                        answer_string=json_line['qas'][index]['answers'][row]['text']
                        if answer_string not in answers:
                            answers.append(answer_string)
                    answer_string=answers[0]
                    answers.remove(answer_string)
                for entity in entities:
                    option=contexts[entity['start']:entity['end']+1]
                    if option not in options and option not in answers:
                        options.append(option.strip().replace("@highlight","").replace("\n", "").replace("\t", "").replace("   ", " ").replace("  ", " "))
                candidates = " ".join([f"({chr(ord('A') + i)}) {x}" for i, x in enumerate(options)])
                contexts=contexts.strip().replace("@highlight","").replace("\n", "").replace("\t", "").replace("   ", " ").replace("  ", " ")
                question=json_line['qas'][index]['query'].replace("@placeholder","_")
                question=question.replace("\n", "").replace("\t", "").replace("   ", " ").replace("  ", " ")
                if '.' not in question:
                    question = question + "." 
                if kind is not "test":     
                    answer_string=answer_string.strip().replace("\n", "").replace("\t", "").replace("   ", " ").replace("  ", " ")
                    answer_index=chr(ord('A') + options.index(answer_string))
                else:
                    answer_string="-"
                    answer_index="-"
                id=json_line['qas'][index]['idx']

                fout.write(f"{question} \\n {candidates} \\n {contexts} \t {answer_string} \n")
                fmeta.write(f"{id}\t{answer_index}\n")
    return counter

def record_mc():
    train_count=record_process_mc("train.jsonl","record","train")
    val_count=record_process_mc("val.jsonl","record","val")
    test_count=record_process_mc("test.jsonl","record","test")
    with open(f"/content/record/counts.json", "w+") as outfile:
        json.dump({"train": train_count, "val": val_count, "test": test_count}, outfile)
        
def quail_process(file,dataset, kind):
    fout = open(f"{dataset}/{kind}.tsv", "w+")
    fmeta = open(f"{dataset}/{kind}_meta.txt", "w+")
    counter=0
    
    with open("/content/quail/quail_v1.3/json/"+file) as f:
        for l in f.readlines()[1:]:
            counter +=1
            json_line = json.loads(l)
            contexts=json_line['context'].strip().replace("\n", "").replace("\t", "").replace("   ", " ").replace("  ", " ")
            options=json_line['answers']
            candidates = " ".join([f"({chr(ord('A') + i)}) {x}" for i, x in enumerate(options)])
            question=json_line['question'].replace("\n", "").replace("\t", "").replace("   ", " ").replace("  ", " ")

            answer_id=int(json_line['correct_answer_id'])
            answer_string=options[answer_id].strip().replace("\n", "").replace("\t", "").replace("   ", " ").replace("  ", " ")
            answer_index=chr(ord('A') + answer_id)
            id=json_line['id']

            fout.write(f"{question} \\n {candidates} \\n {contexts} \t {answer_string} \n")
            fmeta.write(f"{id}\t{answer_index}\n")
    return counter

def quail():
    train_count=quail_process("train.jsonl","quail","train")
    dev_count=quail_process("dev.jsonl","quail","dev")
    challenge_count=quail_process("challenge.jsonl","quail","challenge")
    with open(f"/content/quail/counts.json", "w+") as outfile:
        json.dump({"train": train_count, "dev": dev_count, "challenge": challenge_count}, outfile)
 
def onestopqa_process(dataset, kind):
    fout_adv = open(f"{dataset}_advanced/{kind}.tsv", "w+")
    fmeta_adv = open(f"{dataset}_advanced/{kind}_meta.txt", "w+")
    fout_int = open(f"{dataset}_intermediate/{kind}.tsv", "w+")
    fmeta_int = open(f"{dataset}_intermediate/{kind}_meta.txt", "w+")
    fout_ele = open(f"{dataset}_elementry/{kind}.tsv", "w+")
    fmeta_ele = open(f"{dataset}_elementry/{kind}_meta.txt", "w+")
    counter=0
    paras=[]
    spans=["<A1>", "<A2>", "<A3>", "</A1>", "</A2>", "</A3>", "<D1>", "<D2>", "<D3>", "</D1>", "</D2>", "</D3>", "\n","\t"]
    file_path="/content/onestop-qa/annotations/annotated_articles/*.txt"
    for file in glob.glob(file_path):
        with open(file) as f:
            content=f.read()
            paras.append(content.split("# Paragraph"))
    for paragraph in paras:
        for para in paragraph:
            paralines=para.split("\n")
            paralines=list(filter(None, paralines))
            for idx in range(len(paralines)):
                if paralines[idx].find("Adv: ")!= -1:
                    adv=paralines[idx][5:]
                    adv.strip().replace("   ", " ").replace("  ", " ")
                    for item in spans:
                        adv=adv.replace(item,"")
                if paralines[idx].find("Int: ")!= -1:
                    inter=paralines[idx][5:]
                    inter.strip().replace("   ", " ").replace("  ", " ")
                    for item in spans:
                        inter=inter.replace(item,"")
                if paralines[idx].find("Ele: ")!= -1:
                    ele=paralines[idx][5:]
                    ele.strip().replace("   ", " ").replace("  ", " ")
                    for item in spans:
                        ele=ele.replace(item,"")
                if paralines[idx].find("Q: ")!= -1:
                    counter +=1
                    question=paralines[idx][3:]
                    question=question.strip().replace("\n", "").replace("\t", "").replace("   ", " ").replace("  ", " ")
                    if '?' not in question:
                        question = question + "?"
                    options=paralines[idx+1:idx+5]
                    answer_string=paralines[idx+1][3:]
                    answer_index="A"
                    candidates = " ".join([f"({chr(ord('A') + i)}) {x[3:]}" for i, x in enumerate(options)])

                    fout_adv.write(f"{question} \\n {candidates} \\n {adv} \t {answer_string} \n")
                    fmeta_adv.write(f"{counter}\t{answer_index}\n")
                    fout_int.write(f"{question} \\n {candidates} \\n {inter} \t {answer_string} \n")
                    fmeta_int.write(f"{counter}\t{answer_index}\n")
                    fout_ele.write(f"{question} \\n {candidates} \\n {ele} \t {answer_string} \n")
                    fmeta_ele.write(f"{counter}\t{answer_index}\n")
    return counter

def onestopqa():
    train_count=onestopqa_process("onestopqa","train")
    with open(f"/content/counts.json", "w+") as outfile:
        json.dump({"train": train_count}, outfile)
        
def mcscript_process(file,dataset, kind):
    fout = open(f"{dataset}/{kind}.tsv", "w+")
    fmeta = open(f"{dataset}/{kind}_meta.txt", "w+")
    counter=0

    tree = ET.parse("/content/"+file)
    root = tree.getroot()
    for elem in root:#instance
        context=elem[0].text
        context=context.strip().replace("\n", "").replace("\t", "").replace("--", "").replace("   ", " ").replace("  ", " ")
        id1=elem.get('id')
        for questions in elem[1]:
            counter+=1
            question=questions.get('text')
            question=question.strip().replace("\n", "").replace("\t", "").replace("   ", " ").replace("  ", " ")
            if '?' not in question:
                question = question + "?"
            id2=questions.get('id')
            candidates=[]
            for idx in range(len(questions)):
                if questions[idx].get('correct')=="True":
                    answer_index=chr(ord('A') + idx)
                    answer_string=questions[idx].get('text')
                candidates.append(questions[idx].get('text'))
            options = " ".join([f"({chr(ord('A') + i)}) {x}" for i, x in enumerate(candidates)])
    
            fout.write(f"{question} \\n {options} \\n {context} \t {answer_string} \n")
            fmeta.write(f"{id1} {id2}\t{answer_index}\n")
    return counter

def mcscript():
    train_count=mcscript_process("train-data.xml","mcscript","train")
    dev_count=mcscript_process("dev-data.xml","mcscript","dev")
    test_count=mcscript_process("test-data.xml","mcscript","test")
    with open(f"/content/mcscript/counts.json", "w+") as outfile:
        json.dump({"train": train_count, "dev": dev_count, "test": test_count}, outfile)
    train_count=mcscript_process("train-data.xml","mcscript 2.0","train")
    dev_count=mcscript_process("dev-data.xml","mcscript 2.0","dev")
    test_count=mcscript_process("test-data.xml","mcscript 2.0","test")
    with open(f"/content/mcscript 2.0/counts.json", "w+") as outfile:
        json.dump({"train": train_count, "dev": dev_count, "test": test_count}, outfile)

def adversarialqa():
    for dataset1 in ['dbidaf', 'dbert', 'droberta']:
        dataset = load_dataset("adversarial_qa", dataset1)
        print(f" * dataset: {dataset1}")
        stats = {}
        for split in ['train', 'test', 'dev']:
            outfile = open(f"/Users/danielk/ideaProjects/t2t-qa/t2t-data/adversarialqa_{dataset1}/{split}.tsv", "w+")
            # outfile_meta = open(f"/Users/danielk/ideaProjects/t2t-qa/t2t-data/adversarialqa_{dataset1}/{split}_meta.tsv", "w+")
            split1 = split
            if split == "dev":
                split1 = "validation"
            all_encoded = ""
            # all_encoded_meta = ""
            counter = 0
            for x in dataset[split1]:
                counter += 1
                context = x['context'].lower().replace("\t", " ").replace("\n", " ")
                question = x['question'].lower().replace("\t", " ").replace("\n", " ")
                title = x['title'].lower().replace("\t", " ").replace("\n", " ")
                if len(x['answers']['text']) == 0:
                    answer_text = ""
                else:
                    answer_text = x['answers']['text'][0].lower().replace("\t", " ").replace("\n", " ")
                all_encoded += f"{question} \\n ({title}) {context} \t {answer_text} \n"
                # all_encoded_meta += f"{question} \\n ({title}) {context} \t {answer_text} \n"
            outfile.write(all_encoded)
            stats[split] = counter

        outfile_stat = open(f"/Users/danielk/ideaProjects/t2t-qa/t2t-data/adversarialqa_{dataset1}/counts.json", "w+")
        outfile_stat.write(json.dumps(stats))


def aqua_rat():
    stats = {}
    for split in ['test', 'dev', 'train']:
        file = f"/Users/danielk/ideaProjects/AQuA/{split}.json"
        all_encoded = ""
        outfile = open(f"/Users/danielk/ideaProjects/t2t-qa/t2t-data/aqua_rat/{split}.tsv", "w+")
        counter = 0
        with open(file, "+r") as f:
            for line in f.readlines():
                json_all = json.loads(line.replace("\n", ""))
                print(json_all)
                question = json_all['question'].lower().replace("\n", " ").replace("\t", " ")
                options = json_all['options']
                options_str = " (".join(options).lower().replace("\n", " ").replace("\t", " ")
                options_str = "(" + options_str
                correct = json_all['correct'].lower()
                print(correct)
                correct_idx = ord(correct) - ord('a')
                correct_ans_str = options[correct_idx].split(")")[1]
                all_encoded += f"{question}\\n{options_str}\t{correct_ans_str} \n"
                counter += 1
            outfile.write(all_encoded)
            stats[split] = counter
        outfile_stat = open(f"/Users/danielk/ideaProjects/t2t-qa/t2t-data/aqua_rat/counts.json", "w+")
        outfile_stat.write(json.dumps(stats))


def CODAH():
    file = "/Users/danielk/ideaProjects/CODAH/data/full_data.tsv"
    outfile = open(f"/Users/danielk/ideaProjects/t2t-qa/t2t-data/codah/dev.tsv", "w+")
    counter = 0
    stats = {}
    all_encoded = ""
    with open(file) as f:
        for line in f.readlines():
            counter += 1
            line_split = line.split("\t")
            type = line_split[0]
            question = line_split[1]
            o1 = line_split[2]
            o2 = line_split[3]
            o3 = line_split[4]
            o4 = line_split[5]
            ans_idx = int(line_split[6])
            if ans_idx == 0:
                ans_str = o1
            elif ans_idx == 1:
                ans_str = o2
            elif ans_idx == 2:
                ans_str = o3
            elif ans_idx == 3:
                ans_str = o4
            else:
                raise Exception(f"hm .... {ans_idx}")
            ans_str = ans_str.replace("\n", " ").replace("\t", " ")
            input = f"{question} \\n (a) {o1} (b) {o2} (c) {o3} (d) {o4}".replace("\n", " ").replace("\t", " ")
            all_encoded += f"{input}\t{ans_str} \n"
    outfile.write(all_encoded)
    stats['dev'] = counter
    outfile_stat = open(f"/Users/danielk/ideaProjects/t2t-qa/t2t-data/codah/counts.json", "w+")
    outfile_stat.write(json.dumps(stats))


# COVID-QA: A Question Answering Dataset for COVID-19
# its average context length is around 4k tokens
def covidqa():
    file = "/Users/danielk/Desktop/COVID-QA.json"
    fout = open(f"/Users/danielk/ideaProjects/t2t-qa/t2t-data/covid_qa_deepset/dev.tsv", "w+")
    # ftargets = open(f"/Users/danielk/ideaProjects/t2t-qa/t2t-data/covid_qa_deepset/{segment}_targets.txt", "+w")
    # finputs = open(f"/Users/danielk/ideaProjects/t2t-qa/t2t-data/covid_qa_deepset/{segment}_inputs.txt", "+w")
    ans_size = []
    counter = 0
    stats = {}
    context_size = []
    with open(file) as f:
        file = json.load(f)
        for section in file['data']:
            # title = section['title'].replace("\n", " ").replace("\t", " ")
            for para in section['paragraphs']:
                context = para['context'].replace("\n", " ").replace("\t", " ")
                for qa in para['qas']:
                    question = qa['question'].replace("\n", " ").replace("\t", " ")
                    ans_size.append(len(qa['answers']))
                    for a in qa['answers']:
                        answer = a['text'].replace("\n", " ").replace("\t", " ")
                        fout.write(f"{question} \\n {context}\t{answer}\n")
                        counter += 1
                        context_size.append(len(context.split(" ")))
                        # ftargets.write(f"{answer}\n")
                        # finputs.write(f"{question} \\n ({title}) {context}\n")
    print(sum(ans_size) / len(ans_size))
    stats['dev'] = counter
    outfile_stat = open(f"/Users/danielk/ideaProjects/t2t-qa/t2t-data/covid_qa_deepset/counts.json", "w+")
    outfile_stat.write(json.dumps(stats))
    print(context_size)
    print(statistics.mean(context_size))


def read_and_parse_multiqa(file, dataset, kind):
    fout = open(f"/Users/danielk/ideaProjects/t2t-qa/t2t-data/{dataset}/{kind}.tsv", "w+")
    fmeta = open(f"/Users/danielk/ideaProjects/t2t-qa/t2t-data/{dataset}/{kind}_meta.txt", "w+")
    ans = open(f"/Users/danielk/ideaProjects/t2t-qa/t2t-data/{dataset}/{kind}_ans.jsonl", "w+")

    with open(file) as f:
        for l in f.readlines()[1:]:
            json_line = json.loads(l)
            pid = json_line['id']
            paragraph = ""
            for p in json_line['context']['documents']:
                if 'title' in p:
                    paragraph += f" ({p['title']}) "
                paragraph += p['text']
            paragraph = paragraph.strip().replace("\n", "").replace("\t", "")
            for q in json_line['qas']:
                qid = q['qid']
                fmeta.write(f"{pid}, {qid} \n")
                question = q['question']
                answers = []
                print(q)
                if 'cannot_answer' in q['answers']['open-ended']:
                    if q['answers']['open-ended']['cannot_answer'] == 'yes':
                        answers.append('<No Answer>')
                else:
                    for a in q['answers']['open-ended']['annotators_answer_candidates']:
                        print(a)
                        if 'extractive' in a['single_answer']:
                            answers.append(a['single_answer']['extractive']['answer'])
                        elif 'yesno' in a['single_answer']:
                            answers.append(a['single_answer']['yesno'])
                        else:
                            print("yo yo yo ")

                assert len(answers) > 0

                paragraph = paragraph.replace("\t", "").replace("   ", " ").replace("  ", " ").replace("\n", " ")
                question = question.replace("\t", "").replace("   ", " ").replace("  ", " ").replace("\n", " ")
                if '?' not in question:
                    question = question + "?"
                all_ans = [a.replace("\t", "").replace("   ", " ").replace("  ", " ").replace("\n", " ") for a in
                           answers]

                print(all_ans)
                fout.write(f"{question.strip()} \\n {paragraph.strip()}\t{all_ans[0].strip()}\n")
                ans.write(json.dumps(all_ans) + "\n")


# the average paragraph length is 512 tokens
def duo_rc():
    read_and_parse_multiqa("/Users/danielk/ideaProjects/t2t-qa/datasets/DuoRC_Paraphrase_dev.jsonl",
                           "duo_rc_paraphrase", "dev")
    read_and_parse_multiqa("/Users/danielk/ideaProjects/t2t-qa/datasets/DuoRC_Paraphrase_train.jsonl",
                           "duo_rc_paraphrase", "train")

    read_and_parse_multiqa("/Users/danielk/ideaProjects/t2t-qa/datasets/DuoRC_Self_dev.jsonl", "duo_rc_self", "dev")
    read_and_parse_multiqa("/Users/danielk/ideaProjects/t2t-qa/datasets/DuoRC_Self_train.jsonl", "duo_rc_self", "train")


def headqa():
    def read_file(infile, outfile, outfile_meta):
        outfile = open(outfile, "w+")
        outfile_meta = open(outfile_meta, "w+")
        all_lines = ""
        counter = 0
        with open(infile) as f:
            json_data = json.load(f)
            for ex, v in json_data['exams'].items():
                for x in v['data']:
                    counter += 1
                    print(" - - - - - - - - - ")
                    print(x)
                    question = x['qtext'].lower().replace("\t", " ").replace("\n", " ")
                    answer_idx = int(x['ra']) - 1
                    answers = [y['atext'] for y in x['answers']]
                    candidates = "".join([f" ({chr(ord('A') + i)}) {x}" for i, x in enumerate(answers)])
                    candidates = candidates.lower().replace("\t", " ").replace("\n", " ")
                    correct_ans_str = answers[answer_idx].lower().replace("\t", " ").replace("\n", " ")
                    correct_ans_label = chr(answer_idx + ord('A'))
                    all_lines += f"{question} \\n {candidates} \t {correct_ans_str} \n"
                    outfile_meta.write(f"{x['qid']} \t {correct_ans_label} \n")
        outfile.write(all_lines)
        return counter

    stats = {}
    dir = "/Users/danielk"
    stats['dev'] = read_file(f"{dir}/Desktop/HEAD_EN/dev_HEAD_EN.json",
                             f"{dir}/ideaProjects/t2t-qa/t2t-data/head_qa_en_test/dev.tsv",
                             f"{dir}/ideaProjects/t2t-qa/t2t-data/head_qa_en_test/dev_meta.tsv")
    stats['test'] = read_file(f"{dir}/Desktop/HEAD_EN/test_HEAD_EN.json",
                              f"{dir}/ideaProjects/t2t-qa/t2t-data/head_qa_en_test/test.tsv",
                              f"{dir}/ideaProjects/t2t-qa/t2t-data/head_qa_en_test/test_meta.tsv")
    stats['train'] = read_file(f"{dir}/Desktop/HEAD_EN/train_HEAD_EN.json",
                               f"{dir}/ideaProjects/t2t-qa/t2t-data/head_qa_en_test/train.tsv",
                               f"{dir}/ideaProjects/t2t-qa/t2t-data/head_qa_en_test/train_meta.tsv")

    outfile_stat = open(f"{dir}/ideaProjects/t2t-qa/t2t-data/head_qa_en_test/counts.json", "w+")
    outfile_stat.write(json.dumps(stats))


test_split = [
    1,
    100,
    111,
    113,
    122,
    134,
    136,
    137,
    139,
    144,
    153,
    154,
    157,
    158,
    163,
    164,
    168,
    175,
    179,
    18,
    188,
    19,
    192,
    197,
    202,
    204,
    205,
    206,
    207,
    25,
    28,
    31,
    33,
    34,
    38,
    44,
    50,
    57,
    59,
    66,
    67,
    7,
    71,
    74,
    79,
    84,
    86,
    88,
    97,
    98,
]

dir = "/Users/danielk/Desktop/processbankdata/qa"
def convert_proccess_bank():

    def process(inputfile, outfile):
        all_lines = ""
        counter = 0
        inputfile = open(inputfile, "r")
        outfile = open(outfile, "w+")
        for line in inputfile.readlines():
            counter += 1
            linejson = json.loads(line)
            # print(linejson)
            context = linejson['text'].lower().replace("\t", " ").replace("\n", " ").strip()
            for question in linejson['questions']['question']:
                print(" - - - - ")
                print(question)
                print(type(question))
                if type(question) == str:
                    continue
                q = question['q'].lower().replace("\t", " ").replace("\n", " ").strip()
                a0 = question['a0']
                a1 = question['a1']
                if type(a0) == str:
                    a0 = a0.lower().replace("\t", " ").replace("\n", " ").strip()
                if type(a1) == str:
                    a1 = a1.lower().replace("\t", " ").replace("\n", " ").strip()
                candidates = f"(a) {a0} (b) {a1} ".lower().replace("\t", " ").replace("\n", " ").strip()
                correct = question['correct']
                if correct == 0:
                    correct_str = a0
                else:
                    correct_str = a1

                all_lines += f"{q} \\n {candidates} \\n {context} \t {correct_str} \n"
        outfile.write(all_lines)
        return counter

    dir = "/Users/danielk"
    stats = {}
    stats['test'] = process(f"{dir}/Desktop/processbankdata/test.jsonl", f"{dir}/ideaProjects/t2t-qa/t2t-data/processbank/test.tsv")
    stats['train'] = process(f"{dir}/Desktop/processbankdata/train.jsonl", f"{dir}/ideaProjects/t2t-qa/t2t-data/processbank/train.tsv")
    outfile_stat = open(f"{dir}/ideaProjects/t2t-qa/t2t-data/processbank/counts.json", "w+")
    outfile_stat.write(json.dumps(stats))
    
def cosmosqa_process(file,dataset, kind):
    fout = open(f"{dataset}/{kind}.tsv", "w+")
    fmeta = open(f"{dataset}/{kind}_meta.txt", "w+")
    counter=0
    
    if kind=='test':
        df=pd.read_json("/content/cosmosqa/data/test.jsonl",lines=True)
        questions=df[['id','context','question','answer0','answer1','answer2','answer3']].values
    else:
        df=pd.read_csv("/content/cosmosqa/data/"+file)
        questions=df[['id','context','question','answer0','answer1','answer2','answer3','label']].values

    for row in range(len(questions)):
        id=questions[row][0]
        contexts=questions[row][1].strip().replace("\n", "").replace("\t", "").replace("   ", " ").replace("  ", " ")
        question=questions[row][2].strip().replace("\n", "").replace("\t", "").replace("   ", " ").replace("  ", " ")
        if '?' not in question:
            question = question + "?"
        option1=" (A) "+questions[row][3].strip().replace("\n", "").replace("\t", "").replace("   ", " ").replace("  ", " ")
        option2=" (B) "+questions[row][4].strip().replace("\n", "").replace("\t", "").replace("   ", " ").replace("  ", " ")
        option3=" (C) "+questions[row][5].strip().replace("\n", "").replace("\t", "").replace("   ", " ").replace("  ", " ")
        option4=" (D) "+questions[row][6].strip().replace("\n", "").replace("\t", "").replace("   ", " ").replace("  ", " ")
        options = option1 + option2 + option3 + option4 

        if kind=='test':
            answer="-"
            answer_string="-"
        else:
            answer_index = questions[row][7]
            answer=chr(ord('A')+answer_index)
            answer_string=questions[row][answer_index+3].strip().replace("\n", "").replace("\t", "").replace("   ", " ").replace("  ", " ")
        fmeta.write(f"{id}\t{answer} \n")
        fout.write(f"{question} \\n{options} \\n {contexts}\t{answer_string} \n")
    return len(questions)

def cosmosqa():
    train_count=cosmosqa_process("train.csv","cosmos","train")
    val_count=cosmosqa_process("valid.csv","cosmos","val")
    test_count=cosmosqa_process("test.jsonl","cosmos","test")
    with open(f"/content/cosmos/counts.json", "w+") as outfile:
        json.dump({"train": train_count, "val": val_count, "test": test_count}, outfile)
        
def tweetqa_process(file,dataset, kind):
    fout = open(f"{dataset}/{kind}.tsv", "w+")
    fmeta = open(f"{dataset}/{kind}_meta.txt", "w+")
    df=pd.read_json("/content/"+file)

    if kind=='test':
        questions=df[['Question','Tweet','qid']].values
    else:
        questions=df[['Question','Tweet','qid','Answer']].values

    for row in range(len(questions)):
        id=questions[row][2]
        contexts=questions[row][1].strip().replace("\n", "").replace("\t", "").replace("   ", " ").replace("  ", " ")
        question=questions[row][0].strip().replace("\n", "").replace("\t", "").replace("   ", " ").replace("  ", " ")
        if '?' not in question:
            question = question + "?"
        fmeta.write(f"{id}\n")
        if kind=='test':
            answer_string="-"
        else:
            answer_string=questions[row][3][0].strip().replace("\n", "").replace("\t", "").replace("   ", " ").replace("  ", " ")
        fout.write(f"{question} \\n {contexts}\t{answer_string} \n")
    return len(questions) 

def tweetqa():
    train_count=tweetqa_process("train.json","tweetqa","train")
    dev_count=tweetqa_process("dev.json","tweetqa","dev")
    test_count=tweetqa_process("test.json","tweetqa","test")
    with open(f"/content/tweetqa/counts.json", "w+") as outfile:
        json.dump({"train": train_count, "dev": dev_count, "test": test_count}, outfile)

#measuring massive multitask language understanding dataset
def mmmlu_process(dataset, kind):
    fout = open(f"{dataset}/{kind}.tsv", "w+")
    fmeta = open(f"{dataset}/{kind}_meta.txt", "w+")
    counter=0

    file_path="/content/"+kind+"/*.csv"
    for file in glob.glob(file_path):
        df=pd.read_csv(file,header=None)
        questions=df.values

        for row in range(len(questions)):
            counter+=1
            question=questions[row][0].strip().rstrip("\n").replace("\n", "").replace("\t", "").replace("   ", " ").replace("  ", " ")
            option1=" (A) "+questions[row][1].strip().replace("\n", "").replace("\t", "").replace("   ", " ").replace("  ", " ")
            option2=" (B) "+questions[row][2].strip().replace("\n", "").replace("\t", "").replace("   ", " ").replace("  ", " ")
            option3=" (C) "+questions[row][3].strip().replace("\n", "").replace("\t", "").replace("   ", " ").replace("  ", " ")
            option4=" (D) "+questions[row][4].strip().replace("\n", "").replace("\t", "").replace("   ", " ").replace("  ", " ")
            options = option1 + option2 + option3 + option4 

            answer=questions[row][5]
            answer_index = ord(answer)-ord('A')
            answer_string=questions[row][answer_index+1].strip().replace("\n", "").replace("\t", "").replace("   ", " ").replace("  ", " ")
            fmeta.write(f"{counter}\t{answer} \n")
            fout.write(f"{question} \\n{options} \t{answer_string} \n")
    return counter

def mmmlu():
    val_count=mmmlu_process("measuring_massive_multitask_language_understanding","val")
    dev_count=mmmlu_process("measuring_massive_multitask_language_understanding","dev")
    test_count=mmmlu_process("measuring_massive_multitask_language_understanding","test")
    with open(f"/content/measuring_massive_multitask_language_understanding/counts.json", "w+") as outfile:
        json.dump({"val": val_count, "dev": dev_count, "test": test_count}, outfile)
        
def dream_process(file,dataset, kind):
    fout = open(f"{dataset}/{kind}.tsv", "w+")
    fmeta = open(f"{dataset}/{kind}_meta.txt", "w+")
    counter=0

    df=pd.read_json('/content/dream/data/'+file)
    questions=df.values

    for row in range(len(questions)):
        dialogs=questions[row][0]
        context=" ".join(dialogs)
        context=context.strip().rstrip("\n").replace("\n", "").replace("\t", "").replace("   ", " ").replace("  ", " ")
        materials=questions[row][1]
        id=questions[row][2]
        for item in materials:
            question=item['question'].strip().rstrip("\n").replace("\n", "").replace("\t", "").replace("   ", " ").replace("  ", " ")
            choices=item['choice']
            options = " ".join([f"({chr(ord('A') + i)}) {x}" for i, x in enumerate(choices)])
            answer_string=item['answer'].strip().replace("\n", "").replace("\t", "").replace("   ", " ").replace("  ", " ")
            answer=chr(ord('A') + choices.index(answer_string))
            counter+=1

            fmeta.write(f"{id}\t{answer} \n")
            fout.write(f"{question} \\n{options} \\n {context}\t{answer_string} \n")
    return counter

def dream():
    train_count=dream_process("train.json","Dream","train")
    dev_count=dream_process("dev.json","Dream","dev")
    test_count=dream_process("test.json","Dream","test")
    with open(f"/content/Dream/counts.json", "w+") as outfile:
        json.dump({"train": train_count, "dev": dev_count, "test": test_count}, outfile)

def prost_process(dataset, kind):
    fout_od = open(f"{dataset}_open_domain_with_context/{kind}.tsv", "w+")
    fmeta_od = open(f"{dataset}_open_domain_with_context/{kind}_meta.txt", "w+")
    fout_mc = open(f"{dataset}_multiple_choice_with_context/{kind}.tsv", "w+")
    fmeta_mc = open(f"{dataset}_multiple_choice_with_context/{kind}_meta.txt", "w+")
    fout_od_no = open(f"{dataset}_open_domain_with_no_context/{kind}.tsv", "w+")
    fmeta_od_no = open(f"{dataset}_open_domain_with_no_context/{kind}_meta.txt", "w+")
    fout_mc_no = open(f"{dataset}_multiple_choice_with_no_context/{kind}.tsv", "w+")
    fmeta_mc_no = open(f"{dataset}_multiple_choice_with_no_context/{kind}_meta.txt", "w+")
    counter=0

    dataset =load_dataset('corypaik/prost', split='test')

    for row in range(len(dataset)):
        question=dataset[row]['ex_question'].strip().replace("\n", "").replace("\t", "").replace("   ", " ").replace("  ", " ")
        open_question=dataset[row]['question'].strip().replace("[MASK]","_").replace("\n", "").replace("\t", "").replace("   ", " ").replace("  ", " ")
        context=dataset[row]['context'].strip().rstrip("\n").replace("\n", "").replace("\t", "").replace("   ", " ").replace("  ", " ")
        id=dataset[row]['name']
        choices=[dataset[row]['A'],dataset[row]['B'],dataset[row]['C'],dataset[row]['D']]
        options = " ".join([f"({chr(ord('A') + i)}) {x}" for i, x in enumerate(choices)])
        answer_index=dataset[row]['label']
        answer_string=choices[answer_index]
        answer=chr(ord('A') + answer_index)

        fmeta_mc.write(f"{id}\t{answer} \n")
        fout_mc.write(f"{question} \\n{options} \\n {context}\t{answer_string} \n")
        fmeta_od.write(f"{id}\n")
        fout_od.write(f"{open_question} \\n {context}\t{answer_string} \n")
        fmeta_mc_no.write(f"{id}\t{answer} \n")
        fout_mc_no.write(f"{question} \\n{options}\t{answer_string} \n")
        fmeta_od_no.write(f"{id}\n")
        fout_od_no.write(f"{open_question}\t{answer_string} \n")
    return len(dataset)

def prost():
    test_count=prost_process("prost","test")
    with open(f"/content/counts.json", "w+") as outfile:
        json.dump({"test": test_count}, outfile)

def qaconv():
    mapping = {
        "trn": "train",
        "val": "val", 
        "tst": "test"
    }

    article_json = json.load(open("/content/QAConv/data/article_segment.json"))

    output_dir = "/content/qaconv"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    count_all=[]
    for dtype in ["trn", "val", "tst"]:  
        ques_json = json.load(open("/content/QAConv/data/{}.json".format(dtype)))
        src_all=[]
        meta_all=[]
        count=0
        for qa_pair in ques_json:
            count+=1
            context = article_json[qa_pair["article_segment_id"]]["seg_dialog"]
            context = " ".join(['{}: {}'.format(c["speaker"], c["text"].replace("\n", " ")) for c in context])
            context=context.strip().replace("\n", "").replace("\t", "").replace("   ", " ").replace("  ", " ")
            question=qa_pair["question"].strip().replace("\n", "").replace("\t", "").replace("   ", " ").replace("  ", " ")
            if len(qa_pair["answers"]):
                # here we only use the first potential answers
                answer = qa_pair["answers"][0].strip().replace("\n", "").replace("\t", "").replace("   ", " ").replace("  ", " ")
            else: # unanswerable
                answer = "unanswerable"
            src = "{} \\n {} \t {}".format(question,context,answer)
            src_all.append(src)
            id=qa_pair["id"]
            meta_all.append(id)
        count_all.append(count)
        with open("{}/{}.tsv".format(output_dir, mapping[dtype]), "w") as fout:    
            fout.write("\n".join(src_all))
        with open("{}/{}_meta.txt".format(output_dir, mapping[dtype]), "w") as fout:
            fout.write("\n".join(meta_all))
    with open("{}/counts.json".format(output_dir), "w") as outfile:
        json.dump({"train": count_all[0], "val": count_all[1], "test": count_all[2]}, outfile)
        
anlg()
summarization()
drop()
mctest()
squad()
squad2()
newsqa()
race("string", "high")
race("string", "middle")
boolq()
boolq_np()
searchqa()
arc()
hotpotqa()
narrative_qa()
duo_rc()
quoref()
ropes()
multirc()
openbookqa()
triviaqa()
ai2_science()
commonsenseqa()
qasc()
boolq_contrast_sets()
drop_contrast_sets()
quoref_contrast_sets()
ropes_contrast_sets()
ambigqa()
natural_questions_direct_answer()
winogrande()
physical_iqa()
social_iqa()
csqa()
pubmedqa()
strategyqa()
reclor()
race_c()
record_extractive()
record_mc()
quail()
onestopqa()
mcscript()
convert_proccess_bank()
headqa()
duo_rc()
covidqa()
CODAH()
aqua_rat()
adversarialqa()
cosmosqa()
tweetqa()
mmmlu()
dream()
prost()
qaconv()
