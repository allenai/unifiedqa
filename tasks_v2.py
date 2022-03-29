import t5
import os
import functools
import tensorflow as tf
from t5.data import sentencepiece_vocabulary
from t5.evaluation import metrics

DATA_DIR = "gs://danielk-files/data/"


def get_downloaded_data_path(data_dir1, split, extension):
    return os.path.join(data_dir1, split + extension)

def normalize_text(text):
    """Lowercase and remove quotes from a TensorFlow string."""
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text, "'(.*)'", r"\1")
    return text

def to_inputs_and_targets(ex):
    return {
        "inputs": normalize_text(ex["inputs"]),
        "targets": normalize_text(ex["targets"])
    }

def preprocess(
        dataset,
        prefix='',  # not used
        sample_answer=False,  # not used
):
    return dataset.map(to_inputs_and_targets,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)


def dataset_fn(split, shuffle_files=False, dataset=""):
    # Load lines from the text file as examples.
    ds = tf.data.TextLineDataset(get_downloaded_data_path(DATA_DIR + dataset, split, ".tsv"))
    print(" >>>> about to read tsv . . . ")
    ds = ds.map(
        functools.partial(tf.io.decode_csv, record_defaults=["", "", ""], use_quote_delim=False, field_delim="\t"),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # Map each tuple to a {"question": ... "answers": ...} dict.
    ds = ds.map(lambda *ex: dict(zip(["inputs", "targets", "id"], ex)))
    return ds


def dataset_fn_two_column(split, shuffle_files=False, dataset=""):
    # Load lines from the text file as examples.
    ds = tf.data.TextLineDataset(get_downloaded_data_path(DATA_DIR + dataset, split, ".tsv"))
    print(" >>>> about to read tsv . . . ")
    ds = ds.map(
        functools.partial(tf.io.decode_csv, record_defaults=["", ""], use_quote_delim=False, field_delim="\t"),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # Map each tuple to a {"question": ... "answers": ...} dict.
    ds = ds.map(lambda *ex: dict(zip(["inputs", "targets"], ex)))
    return ds


def postprocessor(answer, example=None, is_target=False):
    """Returns answer, or all answers if the full example is provided."""
    if example:
        return tf.compat.as_text(answer) + "\t" + tf.compat.as_text(example["id"])
    else:
        return answer


def postprocessor_two_column(answer, example=None, is_target=False):
    """Returns answer, or all answers if the full example is provided."""
    return tf.compat.as_text(answer)



for task in [
    "arc_easy_with_ir",
    "arc_hard_with_ir",
    "contrast_sets_boolq",
    "contrast_sets_drop",
    "contrast_sets_quoref",
    "contrast_sets_ropes",
    "race_string",
    "commonsenseqa",
    "arc_hard",
    "arc_easy",
    "mctest_corrected_the_separator",
    "natural_questions",
    "quoref",
    "squad1_1",
    "squad2",
    "boolq",
    "multirc",
    "newsqa",
    "ropes",
    "ropes_test",
    "drop",
    "narrativeqa",
    "openbookqa",
    "qasc",
    "boolq_np",
    "arc_hard_dev",
    "arc_easy_dev",
    "qasc_test",
    "openbookqa_dev",
    "narrativeqa_dev",
    "commonsenseqa_test",
    "qasc_with_ir",
    "qasc_with_ir_test",
    "openbookqa_with_ir",
    "openbookqa_with_ir_dev",
    "arc_easy_with_ir_dev",
    "arc_hard_with_ir_dev",
    "race_string_dev",
    "ambigqa",
    "natural_questions_with_dpr_para",
    "natural_questions_direct_ans_test",

    # new unseen datasets
    "winogrande_xl",
    "social_iqa",
    "social_iqa_test",
    "physical_iqa",
    "physical_iqa_test",
    "adversarialqa_dbert_dev",
    "adversarialqa_dbert_test",
    "adversarialqa_dbidaf_dev",
    "adversarialqa_dbidaf_test",
    "adversarialqa_droberta_dev",
    "adversarialqa_droberta_test",
    "aqua_rat_dev",
    "aqua_rat_test",
    "codah_dev",
    "codah_test",
    "head_qa_en_dev",
    "head_qa_en_test",
    "processbank_test",
    "csqa2",
    "strategyqa",
    'pubmedqa_pqal_short_ans', # only the short answer subset with labeled answers
    'reclor',
    'race_c',
    'quail',
    'onestopqa_elementry',
    'onestopqa_intermediate',
    'onestopqa_advanced',
    'mcscript',
    'mcscript2',
    'record_extractive',
    'record_multiple_choice',
    'cosmosqa',
    'tweetqa',
    'measuring_massive_multitask_language_understanding',
    'dream',
    "qaconv",
]:
    t5.data.TaskRegistry.add(
        f"{task}_mixture",
        # Supply a function which returns a tf.data.Dataset.
        dataset_fn=functools.partial(dataset_fn_two_column, dataset=task),
        splits=["test", "dev"],
        # Supply a function which preprocesses text from the tf.data.Dataset.
        text_preprocessor=preprocess,
        # Lowercase targets before computing metrics.
        postprocess_fn=postprocessor_two_column,
        # sentencepiece_model_path=t5.data.DEFAULT_SPM_PATH,
        metric_fns=[metrics.squad]
    )

# tasks with test set only
for task in [
    "prost_multiple_choice_with_context",
    "prost_multiple_choice_with_no_context",
    "prost_open_domain_with_context",
    "prost_open_domain_with_no_context",
]:
    t5.data.TaskRegistry.add(
        f"{task}_mixture",
        # Supply a function which returns a tf.data.Dataset.
        dataset_fn=functools.partial(dataset_fn_two_column, dataset=task),
        splits=["test"],
        # Supply a function which preprocesses text from the tf.data.Dataset.
        text_preprocessor=preprocess,
        # Lowercase targets before computing metrics.
        postprocess_fn=postprocessor_two_column,
        # sentencepiece_model_path=t5.data.DEFAULT_SPM_PATH,
        metric_fns=[metrics.squad]
    )
    
    
# v2 union model
union_datasets_v2 = [
    "narrativeqa_dev",
    "ai2_science_middle",
    "ai2_science_elementary",
    "arc_hard", "arc_easy",
    "mctest_corrected_the_separator",
    "squad1_1", "squad2",
    "boolq",
    "race_string",
    "openbookqa",
    "quoref",
    "newsqa",
    "ropes",
    "multirc",
    "drop",
    "qasc",
    "boolq_np",
    "commonsenseqa",
    "qasc_with_ir",
    "openbookqa_with_ir",
    "arc_easy_with_ir",
    "arc_hard_with_ir",
    "natural_questions_with_dpr_para",
    "winogrande_xl",
    "social_iqa",
    "physical_iqa",
]
print(f" >>>> adding one mixture for `union_mixture`")
t5.data.MixtureRegistry.add(
    f"union_v2_mixture",
    [f"{d}_task" for d in union_datasets_v2],
    default_rate=1.0
)
    
    
