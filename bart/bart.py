import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import T5ForConditionalGeneration, BartForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput  #TJH added

def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """ TJH: from  modelling_bart.py NOT currently used
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids



class MyBart(BartForConditionalGeneration):
    """ TJH: adding , past_key_values=None to forward(..) takes us to next keyword error 'head_mask'

         Original forward below replaced with new forward (and new =model(...) below)
    def forward(self, input_ids, attention_mask=None, encoder_outputs=None,
            decoder_input_ids=None, decoder_attention_mask=None, decoder_cached_states=None,
            use_cache=False, is_training=False):
        
        New version assumes that for training, decoder inputs are in labels
        and for generation, decoder inputs are in decoder_input_ids
    """
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,  #TJH In 4.4.2 labels contains what in unifiedqa is called decoder_input_ids
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        #TJH: Added for compatibility with 4.4.2
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:  #TJH added for compatibility with other 4.4.2 seq2seq models
            if decoder_input_ids is None:
                #TJH: how it is done in modelling_bart.py. Using the unifiedQA method instead
#                decoder_input_ids = shift_tokens_right(
#                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
#                )
                decoder_start_token_id = self.config.decoder_start_token_id
                decoder_input_ids = labels.new_zeros(labels.shape)
                decoder_input_ids[..., 1:] = labels[..., :-1].clone()
                decoder_input_ids[..., 0] = decoder_start_token_id    
        
        # TJH: below from modeling_bart.py
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,   #TJH: no underscore
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        lm_logits = F.linear(outputs[0], self.model.shared.weight, bias=self.final_logits_bias)
        
        loss = None
        if labels is not None:   #TJH added labels is not None instead of is_training
            loss_fct = nn.CrossEntropyLoss(reduce=False)
            losses = loss_fct(lm_logits.view(-1, self.config.vocab_size),
                              labels.view(-1))
            loss = torch.sum(losses * decoder_attention_mask.float().view(-1))

        if not return_dict: #TJH: from modeling_bart.py
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output    
    
        return Seq2SeqLMOutput( #TJH: from modeling_bart.py. 
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )


    def generate_from_string(self, _input, tokenizer=None, **generator_args):
        assert tokenizer is not None
        if isinstance(_input, str):
            _input = [[0] + tokenizer.encode(_input)]
        if isinstance(_input, list) and isinstance(_input[0], str):
            _input = [[0] + tokenizer.encode(i) for i in _input]
        if isinstance(_input, list):
            if isinstance(_input[0], int):
                _input = [_input]
            _input = torch.LongTensor(_input)
        res = self.generate(_input, **generator_args)
        return ([tokenizer.decode(x, skip_special_tokens=True,
                                  clean_up_tokenization_spaces=True).strip() for x in res])

