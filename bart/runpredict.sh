#run predictions - picks up best_model from output_dir otherwise can specify --checkpoint

python cli.py --do_predict --output_dir /data/thar011/out/unifiedqa_2gputest_from_uqackpt \
        --predict_file /data/thar011/data/unifiedqa/drop/dev.tsv \
        --predict_batch_size 64 \
        --append_another_bos --do_lowercase \
        --verbose \
        --prefix dev_drop_


python cli.py --do_predict --output_dir /data/thar011/out/unifiedqa_2gputest_from_uqackpt \
        --predict_file /data/thar011/data/unifiedqa/ropes/dev.tsv \
        --predict_batch_size 64 \
        --append_another_bos --do_lowercase \
        --verbose \
        --prefix dev_ropes_


