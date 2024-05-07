import os
import copy
import json
import types
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset
from transformers import AutoTokenizer, T5ForConditionalGeneration, T5Config
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput

import utils
from utils import AverageMeter
from abstracts import BaseRunner
from process.span_dropout import dropout_mdd, dropout_dd


class UniGddDataset(Dataset):
    def __init__(self, id_list, context_list, label_list):
        self.id_list = id_list
        self.context_list = context_list
        self.label_list = label_list

    def __len__(self):
        return len(self.context_list)

    def __getitem__(self, index):
        id_ = self.id_list[index]
        context = self.context_list[index]
        label = self.label_list[index]
        return id_, context, label


class UniGddRunner(BaseRunner):
    def __init__(self, args):
        super(UniGddRunner, self).__init__()
        if args.copy:
            model_class = CopyUniGdd
        else:
            model_class = UniGdd

        pretrained_path = f'{args.home}/{args.code}/pretrained/{args.model}/'
        if not os.path.exists(pretrained_path):
            utils.mk_dir(pretrained_path)
            tokenizer = AutoTokenizer.from_pretrained(args.model)
            config = T5Config.from_pretrained(args.model)
            config.dropout_rate = args.dropout
            model = model_class.from_pretrained(args.model, config=config)
            special_tokens = ["<last_turn>", "<user>", "<agent>", "<response>", "<grounding>"]
            tokenizer.add_tokens(special_tokens)
            model.resize_token_embeddings(len(tokenizer))
            tokenizer.save_pretrained(pretrained_path)
            config.save_pretrained(pretrained_path)
            model.save_pretrained(pretrained_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
            config = T5Config.from_pretrained(pretrained_path)
            config.dropout_rate = args.dropout
            model = model_class.from_pretrained(pretrained_path, config=config)
        self.args = args
        self.tokenizer = tokenizer
        self.model = model

    def prepare_dataset(self, args, split='train'):
        id_list, context_list, label_list = [], [], []
        data_map = {
            "dd": {
                "train": "train_all_ranker_results_large",
                "valid": "dev_ranker_results_large"
            },
            "dd-dropout": {
                "train": "train_all_ranker_results_large_dropout",
                "valid": "dev_ranker_results_large"
            },
            "dd-dev-test": {
                "test": "dev_test_ranker_results_large"
            },
            "dd-test": {
                "test": "finaltest_ranker_results_large"
            },
            "mdd": {
                "train": 'multi_all_v1_ranker',
                'valid': 'multi_dev_v1_ranker'
            },
            "mdd-train": {
                "train": 'multi_train_v2_ranker',
                'valid': 'multi_dev_v1_ranker'
            },
            "mdd-dropout": {
                "train": 'multi_all_v1_ranker_dropout',
                'valid': 'multi_dev_v1_ranker'
            },
            "mdd-dev-test": {
                'test': 'multi_dev_test_v1_ranker'
            },
            "mdd-test": {
                'test': 'multi_test_v1_ranker'
            },
            "mdd-unseen-dev-test": {
                'test': 'multi_unseen_dev_test_ranker'
            },
            "mdd-unseen-test": {
                'test': 'multi_unseen_test_ranker'
            },
            "dpr": {
                "train": 'dpr_all',
                'valid': 'dpr_dev'
            },
            "dpr-train": {
                "train": 'dpr_train',
                'valid': 'dpr_dev'
            },
            "dpr-dev-test": {
                "test": 'dpr_dev_test'
            },
            "dpr-test": {
                "test": 'dpr_test'
            },
            "dpr-unseen-dev-test": {
                "test": 'dpr_unseen_dev_test'
            },
            "dpr-unseen-test": {
                "test": 'dpr_unseen_test'
            },
        }

        with open(f'{args.home}/g4/{data_map[args.dataset][split]}.json', 'r') as f:
            dial_data = json.load(f)
        for dial in dial_data:
            id_ = dial['id']
            questions = dial['question']
            if isinstance(dial['ctxs'][0], dict):
                passages = [ctx['text'] for ctx in dial['ctxs']]
            else:
                passages = dial['ctxs']
            grounding = dial['grounding']
            response = dial['response']
            if args.data_type == 'all':
                source_line1 = 'generate <grounding> then <response>: ' + ' '.join(questions) + ' '.join(passages)
                tgt_line1 = f"<grounding> {''.join(grounding)} <response> {response}"
            elif args.data_type == 'grounding':
                source_line1 = 'generate <grounding>: ' + ' '.join(questions) + ' '.join(passages)
                tgt_line1 = f"<grounding> {''.join(grounding)}"
            elif args.data_type == 'response':
                source_line1 = 'generate <response>: ' + ' '.join(questions) + ' '.join(passages)
                tgt_line1 = f"<response> {response}"
            elif args.data_type == 'reverse':
                source_line1 = 'generate <response> then <grounding>: ' + ' '.join(questions) + ' '.join(passages)
                tgt_line1 = f"<response> {response} <grounding> {''.join(grounding)} "
            else:
                raise Exception('data type undefined')
            id_list.append(id_)
            context_list.append(source_line1)
            label_list.append(tgt_line1)

        # create dataset
        dataset = UniGddDataset(id_list, context_list, label_list)
        return dataset

    def reload_train_dataset_per_epoch(self, args, epoch):
        datapath = f'{args.home}/g4'
        id_list, context_list, label_list = [], [], []
        if 'mdd' in args.dataset:
            dial_data = dropout_mdd(datapath)
        else:
            dial_data = dropout_dd(datapath)
        for dial in dial_data:
            id_ = dial['id']
            questions = dial['question']
            if isinstance(dial['ctxs'][0], dict):
                passages = [ctx['text'] for ctx in dial['ctxs']]
            else:
                passages = dial['ctxs']
            grounding = dial['grounding']
            response = dial['response']
            if args.data_type == 'all':
                source_line1 = 'generate <grounding> then <response>: ' + ' '.join(questions) + ' '.join(passages)
                tgt_line1 = f"<grounding> {''.join(grounding)} <response> {response}"
            elif args.data_type == 'grounding':
                source_line1 = 'generate <grounding>: ' + ' '.join(questions) + ' '.join(passages)
                tgt_line1 = f"<grounding> {''.join(grounding)}"
            elif args.data_type == 'response':
                source_line1 = 'generate <response>: ' + ' '.join(questions) + ' '.join(passages)
                tgt_line1 = f"<response> {response}"
            else:
                raise Exception('data type undefined')
            id_list.append(id_)
            context_list.append(source_line1)
            label_list.append(tgt_line1)
        # create dataset
        dataset = UniGddDataset(id_list, context_list, label_list)
        return dataset

    def forward(self, model, payload, tokenizer, device):
        ids, context, label = payload
        tokenizer_outputs = tokenizer.batch_encode_plus(list(context), padding=True, return_tensors='pt',
                                                        max_length=self.args.source_sequence_size, truncation=True)
        input_ids = tokenizer_outputs.input_ids.to(device)
        attention_mask = tokenizer_outputs.attention_mask.to(device)
        label_ids = tokenizer.batch_encode_plus(
            list(label), padding=True, return_tensors='pt', max_length=self.args.target_sequence_size,
            truncation=True).input_ids.to(device)
        outputs = model(input_ids=input_ids, labels=label_ids, attention_mask=attention_mask)
        return outputs

    def inference(self, model, payload, tokenizer, device, result_dict):
        ids, context, label = payload
        tokenizer_outputs = tokenizer.batch_encode_plus(list(context), padding=True, return_tensors='pt',
                                                        max_length=self.args.source_sequence_size, truncation=True)
        input_ids = tokenizer_outputs.input_ids.to(device)
        attention_mask = tokenizer_outputs.attention_mask.to(device)
        outputs = model.generate(input_ids, num_beams=self.args.beam_size, max_length=self.args.target_sequence_size,
                                 attention_mask=attention_mask)
        hypothesis = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        result_dict['id_list'] += ids
        result_dict['outputs'] += hypothesis
        result_dict['targets'] += label

    def init_losses_dict(self):
        losses = {
            'loss': AverageMeter('Loss', ':.4e'),
        }
        return copy.deepcopy(losses)

    def measure_loss(self, payload, model_output, losses):
        ids, context, label = payload
        loss = model_output[0]
        losses['loss'].update(loss.item(), len(context))

    def init_results_dict(self):
        results = {
            'id_list': [],
            'outputs': [],
            'targets': []
        }
        return copy.deepcopy(results)

    def init_meters_dict(self):
        meters = {
            'f1': AverageMeter('F1_U', ':6.4f')
        }
        return copy.deepcopy(meters)

    def measure_result(self, result_dict, meters):
        hypothesis_list = [x.split('<response>')[-1].strip() for x in result_dict['outputs']]
        hypothesis_list = [x if x else '@' for x in hypothesis_list]
        reference_list = [x.split('<response>')[-1].strip() for x in result_dict['targets']]
        instance_num = len(hypothesis_list)

        # F1
        f1, em = utils.matching_evaluate(reference_list, hypothesis_list)
        meters['f1'].update(f1, instance_num)


class UniGdd(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.overwrite_forward_crossattention()
        self.config = config

    def overwrite_forward_crossattention(self):
        """
        Replace cross-attention forward function
        """
        for i, mod in enumerate(self.decoder.block):
            attn = mod.layer[1].EncDecAttention
            attn.forward = types.MethodType(cross_attention_forward, attn)


class CopyUniGdd(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.overwrite_forward_crossattention()
        self.config = config
        self.prob_proj = nn.Linear(config.d_model * 2, 1)
        self.sig_proj = nn.Sigmoid()
        self.gen_softmax = nn.Softmax(dim=-1)
        self.copy_softmax = nn.Softmax(dim=-1)

    def overwrite_forward_crossattention(self):
        """
        Replace cross-attention forward function
        """
        for i, mod in enumerate(self.decoder.block):
            attn = mod.layer[1].EncDecAttention
            attn.forward = types.MethodType(cross_attention_forward, attn)

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.BoolTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            decoder_head_mask: Optional[torch.FloatTensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=True,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)
        # copy
        memory = encoder_outputs.last_hidden_state
        decode_output = decoder_outputs.last_hidden_state

        decode_attn = decoder_outputs.cross_attentions[-1]
        decode_attn = torch.mean(decode_attn, dim=1)
        # steps表示真值数量，seq表示原始文本数量，都是batch_size中最大值
        batch_size, steps, seq = decode_attn.size()
        src = input_ids.unsqueeze(1).repeat([1, steps, 1])
        # vocab
        gen_logits = lm_logits
        copy_logits = torch.zeros_like(gen_logits)
        context = torch.matmul(decode_attn, memory)
        copy_logits = copy_logits.scatter_add(2, src, decode_attn)
        prob = self.sig_proj(self.prob_proj(torch.cat([context, decode_output], -1)))

        gen_logits = prob * gen_logits
        copy_logits = (1 - prob) * copy_logits
        lm_logits = gen_logits + copy_logits

        # gen_logits = prob * self.gen_softmax(gen_logits)
        # copy_logits = (1 - prob) * self.copy_softmax(copy_logits)
        # lm_logits = torch.log(gen_logits + copy_logits)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # loss_fct = NLLLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


def cross_attention_forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
):
    # Input is (batch_size, seq_length, dim)
    # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
    # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
    batch_size, seq_length = hidden_states.shape[:2]

    real_seq_length = seq_length

    if past_key_value is not None:
        assert (
                len(past_key_value) == 2
        ), f"past_key_value should have 2 past states: keys and values. Got {len(past_key_value)} past states"
        real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

    key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

    def shape(states):
        """projection"""
        return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

    def unshape(states):
        """reshape"""
        return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

    def project(hidden_states, proj_layer, key_value_states, past_key_value):
        """projects hidden states correctly to key/query states"""
        if key_value_states is None:
            # self-attn
            # (batch_size, n_heads, seq_length, dim_per_head)
            hidden_states = shape(proj_layer(hidden_states))
        elif past_key_value is None:
            # cross-attn
            # (batch_size, n_heads, seq_length, dim_per_head)
            hidden_states = shape(proj_layer(key_value_states))

        if past_key_value is not None:
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, key_length, dim_per_head)
                hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
            else:
                # cross-attn
                hidden_states = past_key_value
        return hidden_states

    # get query states
    query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

    # get key/value states
    key_states = project(
        hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
    )
    value_states = project(
        hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
    )

    # compute scores
    scores = torch.matmul(
        query_states, key_states.transpose(3, 2)
    )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

    if position_bias is None:
        if not self.has_relative_attention_bias:
            position_bias = torch.zeros(
                (1, self.n_heads, real_seq_length, key_length), device=scores.device, dtype=scores.dtype
            )
            if self.training and self.gradient_checkpointing:
                position_bias.requires_grad = True
        else:
            position_bias = self.compute_bias(real_seq_length, key_length)

        # if key and values are already calculated
        # we want only the last query position bias
        if past_key_value is not None:
            position_bias = position_bias[:, :, -hidden_states.size(1):, :]

        if mask is not None:
            position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)

    scores += position_bias
    attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
        scores
    )  # (batch_size, n_heads, seq_length, key_length)
    attn_weights = nn.functional.dropout(
        attn_weights, p=self.dropout, training=self.training
    )  # (batch_size, n_heads, seq_length, key_length)

    # Mask heads if we want to
    if layer_head_mask is not None:
        attn_weights = attn_weights * layer_head_mask

    attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    attn_output = self.o(attn_output)

    present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None
    outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

    if output_attentions:
        outputs = outputs + (attn_weights,)
    return outputs
