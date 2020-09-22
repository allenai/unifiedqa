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
