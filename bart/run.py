import os
import numpy as np
from tqdm import tqdm

import torch
from transformers import BartTokenizer, BartConfig
from transformers import AdamW, get_linear_schedule_with_warmup

from data import QAData
from unified_data import UnifiedQAData
from bart import MyBart

def run(args, logger):
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")  #TJH: bart-large

    if args.is_unifiedqa:
        dev_data = UnifiedQAData(logger, args, args.predict_file, False)
    else:
        dev_data = QAData(logger, args, args.predict_file, False)

    if not args.skip_inference:
        dev_data.load_dataset(tokenizer)
        dev_data.load_dataloader()

    if args.do_train:
        if args.is_unifiedqa:
            train_data = UnifiedQAData(logger, args, args.train_file, True)
        else:
            train_data = QAData(logger, args, args.train_file, True)
        train_data.load_dataset(tokenizer)
        train_data.load_dataloader()

        if args.checkpoint is not None:
            model = MyBart.from_pretrained("facebook/bart-large",
                                           state_dict=torch.load(args.checkpoint))  #TJH: bart-large
            logger.info("Loading checkpoint from {}".format(args.checkpoint))       #TJH Added
        else:
            model = MyBart.from_pretrained("facebook/bart-large") #TJH: bart-large
        if args.n_gpu>1:
            model = torch.nn.DataParallel(model)
        if args.n_gpu>0:
            model.to(torch.device("cuda"))

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler =  get_linear_schedule_with_warmup(optimizer,
                                        num_warmup_steps=args.warmup_steps,
                                        num_training_steps=100000)
        train(args, logger, model, train_data, dev_data, optimizer, scheduler)

    if args.do_predict:
        checkpoint = os.path.join(args.output_dir, 'best-model.pt') if args.checkpoint is None else args.checkpoint
        model = MyBart.from_pretrained("facebook/bart-large",
                                       state_dict=torch.load(checkpoint)) #TJH: bart-large
        logger.info("Loading checkpoint from {}".format(checkpoint))
        if args.n_gpu>0:
            model.to(torch.device("cuda"))
        model.eval()
        ems = inference(model, dev_data, save_predictions=True)
        logger.info("%s on %s data: %.2f" % (dev_data.metric, dev_data.data_type, np.mean(ems)*100))

def train(args, logger, model, train_data, dev_data, optimizer, scheduler):
    model.train()
    global_step = 0
    train_losses = []
    best_accuracy = -1
    stop_training=False

    if args.checkpoint_step > 0:
        for _ in range(args.checkpoint_step):
            global_step += 1
            scheduler.step()

    def convert_to_single_gpu(state_dict):
        def _convert(key):
            if key.startswith('module.'):
                return key[7:]
            return key
        return {_convert(key):value for key, value in state_dict.items()}

    logger.info("Starting training!")
    for epoch in range(int(args.num_train_epochs)):
        if args.verbose: 
            logger.info("Starting Epoch %d" % (epoch))  #TJH added
        for batch in train_data.dataloader:
            if args.verbose and global_step % 100 == 0:
                logger.info("Epoch %d   Global Step %d" % (epoch, global_step))   #TJH Added
            global_step += 1
            batch = [b.to(torch.device("cuda")) for b in batch]
# TJH: this was the original unifiedqa:            
#            loss = model(input_ids=batch[0], attention_mask=batch[1],
#                         decoder_input_ids=batch[2], decoder_attention_mask=batch[3],
#                         is_training=True)
            outputs = model(input_ids=batch[0], attention_mask=batch[1],
                         labels=batch[2], decoder_attention_mask=batch[3])  
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]  #TJH added 
            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
            if torch.isnan(loss).data:
                logger.info("Stop training because loss=%s" % (loss.data))
                stop_training=True
                break
            train_losses.append(loss.detach().cpu())
            loss.backward()

            if global_step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()    # We have accumulated enought gradients
                scheduler.step()
                model.zero_grad()

            if global_step % args.eval_period == 0:
                if args.skip_inference:
                    logger.info("Step %d (epoch %d) Train loss %.2f" % (
                            global_step,
                            epoch,
                            np.mean(train_losses)))
                    train_losses = []
                    model_state_dict = {k:v.cpu() for (k, v) in model.state_dict().items()}
                    if args.n_gpu > 1:
                        model_state_dict = convert_to_single_gpu(model_state_dict)
                    torch.save(model_state_dict, os.path.join(args.output_dir,
                                                              "best-model-{}.pt".format(str(global_step).zfill(6))))
                else:
                    if args.verbose:
                        logger.info("Step %d Starting inference.." % (global_step)) #TJH Added
                    model.eval()
                    curr_em = inference(model if args.n_gpu==1 else model.module, dev_data)
                    logger.info("Step %d Train loss %.2f %s %.2f%% on epoch=%d" % (
                            global_step,
                            np.mean(train_losses),
                            dev_data.metric,
                            curr_em*100,
                            epoch))
                    train_losses = []
                    if best_accuracy < curr_em:
                        model_state_dict = {k:v.cpu() for (k, v) in model.state_dict().items()}
                        if args.n_gpu > 1:
                            model_state_dict = convert_to_single_gpu(model_state_dict)
                        torch.save(model_state_dict, os.path.join(args.output_dir, "best-model.pt"))
                        logger.info("Saving model with best %s: %.2f%% -> %.2f%% on epoch=%d, global_step=%d" % \
                                (dev_data.metric, best_accuracy*100.0, curr_em*100.0, epoch, global_step))
                        best_accuracy = curr_em
                        wait_step = 0
                        stop_training = False
                    else:
                        wait_step += 1
                        logger.info("No improvement. Number of wait steps: %d of max wait steps: %d" % (wait_step, args.wait_step))
                        if wait_step >= args.wait_step:
                            stop_training = True
                            logger.info("Early Stopping due to no improvement after %d wait steps!" % (wait_step))   #TJH Added
                            break
                model.train()
        if stop_training:
            break

def inference(model, dev_data, save_predictions=False):
    predictions = []
    bos_token_id = dev_data.tokenizer.bos_token_id
    if dev_data.args.verbose:
        dev_data.dataloader = tqdm(dev_data.dataloader)
    for i, batch in enumerate(dev_data.dataloader):
        batch = [b.to(torch.device("cuda")) for b in batch]
        outputs = model.generate(input_ids=batch[0],
                                 attention_mask=batch[1],
                                 num_beams=dev_data.args.num_beams,
                                 min_length=1,  #TJH: was min_lnegth
                                 max_length=dev_data.args.max_output_length,
                                 early_stopping=True,)
        for input_, output in zip(batch[0], outputs):
            pred = dev_data.decode(output)
            predictions.append(pred)
    if save_predictions:
        dev_data.save_predictions(predictions)
    return np.mean(dev_data.evaluate(predictions))







