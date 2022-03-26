"""
The official evaluation script for TweetQA, the normalize function is based on the evaluation of SQuAD
"""

import string
import re
import json

# import nltk
# nltk.download()


from nltk.translate.bleu_score import sentence_bleu
import numpy as np

from nlgeval.pycocoevalcap.meteor.meteor import Meteor
from nlgeval.pycocoevalcap.rouge.rouge import Rouge


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

meteor_scorer = Meteor()
rouge_scorer = Rouge()

def ans_score(ans, gold_list):
    ans = normalize_answer(ans)
    gold_list = [normalize_answer(ref) for ref in gold_list]
    bleu = sentence_bleu([_.split() for _ in gold_list], ans.split(), weights=(1,0,0,0))
    # meteor, _ = meteor_scorer.compute_score({0:gold_list}, {0:[ans]})
    # rouge, _ = rouge_scorer.compute_score({0:gold_list}, {0:[ans]})
    # Daniel (December'21): disabled metrics other than BLEU since we don't use them (and they seem to correlate)
    return {
        'bleu': bleu,
        # 'meteor':meteor,
        # 'rouge': rouge
    }

def evaluate(test_annotation_file, user_annotation_file, phase_codename, **kwargs):
    gold_file = test_annotation_file
    pred_file = user_annotation_file
    gold = json.load(open(gold_file))
    pred = json.load(open(pred_file))
    idx2gold = {item['qid']:item['Answer'] for item in gold}
    idx2pred = {item['qid']:item['Answer'] for item in pred}
    idx2scores = {}
    for id_ in idx2gold.keys():
        if isinstance(idx2pred[id_], list):
            pred_ans = idx2pred[id_][0]
        else:
            pred_ans = idx2pred[id_]
        idx2scores[id_] = ans_score(pred_ans, idx2gold[id_])
    bleus = [item['bleu'] for item in idx2scores.values()]
    meteors = [item['meteor'] for item in idx2scores.values()]
    rouges = [item['rouge'] for item in idx2scores.values()]
    print({'BLEU': np.mean(bleus), 'METEOR': np.mean(meteors), 'ROUGE': np.mean(rouges)})

    output = {}
    output['result'] = [
    {'test_split': 
        {
        'BLEU-1': np.mean(bleus),
        'METEOR': np.mean(meteors),
        'ROUGE': np.mean(rouges)
        }
    }
    ]

    return output

# if __name__ == '__main__':
#     pred_file = sys.argv[1]
#     gold_file = sys.argv[2]
#     eval(pred_file, gold_file)

