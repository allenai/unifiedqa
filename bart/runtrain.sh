# training command
# other options:  --do_predict --skip_inference --debug --checkpoint ${unifiedqa_checkpoint}
# --prefix dev_  --prefix test_

python cli.py --do_train --output_dir /data/thar011/out/unifiedqa \
        --is_unifiedqa \
        --checkpoint /data/thar011/ckpts/unifiedqa-bart-large-allenai/unifiedQA-uncased/best-model.pt \
        --train_file /data/thar011/data/unifiedqa/train.tsv \
        --predict_file /data/thar011/data/unifiedqa/dev.tsv \
        --train_batch_size 64 \
        --predict_batch_size 64 \
        --append_another_bos --do_lowercase \
        --eval_period 10000 --verbose \
        --num_train_epochs 10000 \
        --gradient_accumulation_steps 1

