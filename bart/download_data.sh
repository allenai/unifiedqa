#!/bin/sh


BASE_URL=https://storage.googleapis.com/danielk-files/data

if [ $1 = "pretrained" ]

then
    echo "Downloading pre-training datasets"
    mkdir -p pretrain-data
    # datasets for pretraining
    for dataset in narrativeqa ai2_science_middle ai2_science_elementary arc_hard arc_easy mctest_corrected_the_separator squad1_1 squad2 boolq race_string openbookqa ; do
        mkdir -p pretrain-data/${dataset}
        for data_type in train dev test ; do
            wget ${BASE_URL}/${dataset}/${data_type}.tsv -O pretrain-data/${dataset}/${data_type}.tsv
        done
    done

else
    echo "Downloading other datasets"
    mkdir -p data
    # other datasets
    for dataset in qasc qasc_with_ir commonseseqa openbookqa_with_ir arc_hard_with_ir arc_easy_with_ir winogrande_xl physical_iqa social_iqa ropes natural_questions_with_dpr_para ; do
        mkdir -p data/${dataset}
        for data_type in train dev test ; do
            wget ${BASE_URL}/${dataset}/${data_type}.tsv -O data/${dataset}/${data_type}.tsv
        done
    done

fi

exit

if [ $1 = "pretrained" ];

then
    wget https://nlp.cs.washington.edu/ambigqa/data/nqopen.zip -O pretrain-data/nqopen.zip
    unzip -d pretrain-data pretrain-data/nqopen.zip
    rm pretrain-data/nqopen.zip

else

    wget https://nlp.cs.washington.edu/ambigqa/data/nqopen.zip -O data/nqopen.zip
    unzip -d data data/nqopen.zip
    rm data/nqopen.zip
