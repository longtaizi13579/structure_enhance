import os
import copy
import json
import types
import sacrebleu
from rouge import Rouge

import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer, T5ForConditionalGeneration

import utils
from utils import AverageMeter
from abstracts import BaseRunner


class UniverseUniGddDataset(Dataset):
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


def text2line(text):
    return text.replace("\n", " ").replace("\r", " ").strip()


def btag(tag, text):  # tag the content
    return "<{}>{}</{}>".format(tag, text2line(text), tag)


class UniverseUniGddRunner(BaseRunner):
    def __init__(self, args):
        super(UniverseUniGddRunner, self).__init__()
        model_name = 't5-base'
        pretrained_path = f'{args.home}/{args.code}/pretrained/{model_name}/'
        if not os.path.exists(pretrained_path):
            utils.mk_dir(pretrained_path)
            tokenizer = AutoTokenizer.from_pretrained('t5-base')
            model = UniGdd.from_pretrained('t5-base')
            special_tokens = ["<last_turn>", "<user>", "<agent>", "<response>", "<grounding>"]
            tokenizer.add_tokens(special_tokens)
            model.resize_token_embeddings(len(tokenizer))
            model.save_pretrained(f'{pretrained_path}/model')
            tokenizer.save_pretrained(f'{pretrained_path}/tokenizer')
        else:
            tokenizer = AutoTokenizer.from_pretrained(f'{pretrained_path}/tokenizer')
            model = UniGdd.from_pretrained(f'{pretrained_path}/model')
        self.tokenizer = tokenizer
        self.model = model

    def prepare_dataset(self, args, split='train'):
        id_list, context_list, label_list = [], [], []

        if split == 'train':
            # multidoc2dial
            passages = f'{args.home}/g4/mdd_passages.jsonl'
            retrieved = f'{args.home}/g4/train.jsonl'
            multi_dial_data = []
            with open(f'{args.home}/multidoc2dial/multidoc2dial_dial_train.json', 'r') as f:
                for k, v in json.load(f)['dial_data'].items():
                    multi_dial_data += v

            # doc2dial
            dial_data = []
            with open(f'{args.home}/doc2dial/doc2dial_dial_train.json', 'r') as f:
                for domain, doc_dial in json.load(f)['dial_data'].items():
                    for k, v in doc_dial.items():
                        dial_data += v

        elif split == 'valid':
            # multidoc2dial
            passages = f'{args.home}/g4/mdd_passages.jsonl'
            retrieved = f'{args.home}/g4/dev.jsonl'
            multi_dial_data = []
            with open(f'{args.home}/multidoc2dial/multidoc2dial_dial_validation.json', 'r') as f:
                for k, v in json.load(f)['dial_data'].items():
                    multi_dial_data += v

            # doc2dial
            dial_data = []
            with open(f'{args.home}/doc2dial/doc2dial_dial_validation.json', 'r') as f:
                for domain, doc_dial in json.load(f)['dial_data'].items():
                    for k, v in doc_dial.items():
                        dial_data += v

        elif split == 'test':
            with open(f'{args.home}/g4/unseen_dev_test.json', 'r') as f:
                data = json.load(f)
                for sample in data:
                    id_list.append(sample['id'])
                    context = 'generate <grounding> then <response>: ' + ' '.join(sample['question']) + ' '.join(
                        sample['ctxs'])
                    context_list.append(context)
                    label_list.append(sample['response'])
            # create dataset
            dataset = UniverseUniGddDataset(id_list, context_list, label_list)
            return dataset
        else:
            raise Exception(f'Undefined Split')

        # multidoc2dial
        id_span_map = {}
        with open(f'{args.home}/multidoc2dial/multidoc2dial_doc.json', 'r') as f:
            doc_data = json.load(f)['doc_data']
            for domain, docs in doc_data.items():
                for doc_id, content in docs.items():
                    for span_id, span in content['spans'].items():
                        id_span_map[f'{doc_id}_{span_id}'] = span['text_sp']

        id_psg_map = {}
        with open(passages, 'r') as f:
            for line in f.readlines():
                sample = json.loads(line)
                id_psg_map[sample['pid']] = sample['text']

        id_retrieve_map = {}
        with open(retrieved, 'r') as f:
            for line in f.readlines():
                sample = json.loads(line)
                id_retrieve_map[sample['id']] = [x[0] for x in sample['scored_pids']]

        for dial in multi_dial_data:
            dialog_id = dial['dial_id']
            turns = dial["turns"]
            all_prev_utterances = []
            for i, turn in enumerate(turns):
                utterance = turn["utterance"].replace("\n", " ").replace("\r", " ").replace("\t", " ").strip()
                all_prev_utterances.append("<{}> {}".format(turn["role"], utterance))

                if turn["role"] == "agent":
                    continue
                if turn['turn_id'] < len(turns):
                    next_turn = turns[turn['turn_id']]
                    if not (
                            next_turn["role"] == "agent"
                            and next_turn["da"] != "respond_no_solution"
                    ):
                        continue
                else:
                    continue

                id_ = dialog_id + '_' + str(next_turn['turn_id'])
                question = list(reversed(all_prev_utterances))
                question[0] = question[0].replace('<user> ', '<last_turn>')
                response = next_turn['utterance']
                grounding = [id_span_map[f"{x['doc_id']}_{x['id_sp']}"] for x in next_turn['references']]
                passages = [id_psg_map[x] for x in id_retrieve_map[id_]]
                content = ' '.join(question) + ' '.join(passages)
                task_prompt = 'multiple documents dialog ==> '

                if split == 'train':
                    source_line = task_prompt + 'generate <grounding> then <response>: ' + content
                    tgt_line = f"<grounding> {''.join(grounding)} <response> {response}"
                    id_list.append(id_)
                    context_list.append(source_line)
                    label_list.append(tgt_line)

                    source_line = task_prompt + 'generate <grounding>: ' + content
                    tgt_line = f"<grounding> {''.join(grounding)}"
                    id_list.append(id_)
                    context_list.append(source_line)
                    label_list.append(tgt_line)

                    source_line = task_prompt + 'generate <response>: ' + content
                    tgt_line = f"<response> {response}"
                    id_list.append(id_)
                    context_list.append(source_line)
                    label_list.append(tgt_line)
                else:
                    source_line = task_prompt + 'generate <grounding> then <response>: ' + content
                    tgt_line = f"<grounding> {''.join(grounding)} <response> {response}"
                    id_list.append(id_)
                    context_list.append(source_line)
                    label_list.append(tgt_line)

        # doc2dial
        doc_dataset = []
        with open(f'{args.home}/doc2dial/doc2dial_doc.json', 'r') as f:
            doc_data = json.load(f)['doc_data']
            for domain, docs in doc_data.items():
                for k, v in docs.items():
                    doc_dataset.append(v)

        id_doc_map = {}
        for ex in doc_dataset:
            doc_id = ex["doc_id"]
            doc_title = btag("title", ex["title"].split("#")[0])
            spans_text = []
            for d_span in ex["spans"].values():
                tag = d_span["tag"]
                text_sp = d_span["text_sp"]

                if tag != "u":
                    spans_text.append(btag(tag, text2line(text_sp)))
                else:
                    spans_text.append(text2line(text_sp))
            id_doc_map[doc_id] = " ".join([doc_title] + spans_text)

        for dial in dial_data:
            doc_id = dial['doc_id']
            dialog_id = dial['dial_id']
            turns = dial["turns"]
            all_prev_utterances = []
            for i, turn in enumerate(turns):
                utterance = turn["utterance"].replace("\n", " ").replace("\r", " ").replace("\t", " ").strip()
                all_prev_utterances.append("<{}> {}".format(turn["role"], utterance))

                if turn["role"] == "agent":
                    continue
                if turn['turn_id'] < len(turns):
                    next_turn = turns[turn['turn_id']]
                    if not (
                            next_turn["role"] == "agent"
                            and next_turn["da"] != "respond_no_solution"
                    ):
                        continue
                else:
                    continue

                id_ = dialog_id + '_' + str(next_turn['turn_id'])
                question = list(reversed(all_prev_utterances))
                question[0] = question[0].replace('<user> ', '<last_turn>')
                response = next_turn['utterance']
                grounding = [id_span_map[f"{doc_id}_{x['sp_id']}"] for x in next_turn['references']]
                passages = id_doc_map[doc_id]
                content = ' '.join(question) + ' '.join(passages)
                task_prompt = 'single document dialog ==> '

                if split == 'train':
                    source_line = task_prompt + 'generate <grounding> then <response>: ' + content
                    tgt_line = f"<grounding> {''.join(grounding)} <response> {response}"
                    id_list.append(id_)
                    context_list.append(source_line)
                    label_list.append(tgt_line)

                    source_line = task_prompt + 'generate <grounding>: ' + content
                    tgt_line = f"<grounding> {''.join(grounding)}"
                    id_list.append(id_)
                    context_list.append(source_line)
                    label_list.append(tgt_line)

                    source_line = task_prompt + 'generate <response>: ' + content
                    tgt_line = f"<response> {response}"
                    id_list.append(id_)
                    context_list.append(source_line)
                    label_list.append(tgt_line)
                else:
                    source_line = task_prompt + 'generate <grounding> then <response>: ' + content
                    tgt_line = f"<grounding> {''.join(grounding)} <response> {response}"
                    id_list.append(id_)
                    context_list.append(source_line)
                    label_list.append(tgt_line)
        # create dataset
        dataset = UniverseUniGddDataset(id_list, context_list, label_list)
        return dataset

    def forward(self, model, payload, tokenizer, device):
        ids, context, label = payload
        input_ids = tokenizer.batch_encode_plus(
            list(context), padding=True, return_tensors='pt', max_length=2560, truncation=True).input_ids.to(device)
        label_ids = tokenizer.batch_encode_plus(
            list(label), padding=True, return_tensors='pt', max_length=500, truncation=True).input_ids.to(device)
        outputs = model(input_ids=input_ids, labels=label_ids)
        return outputs

    def inference(self, model, payload, tokenizer, device, result_dict):
        ids, context, label = payload
        input_ids = tokenizer.batch_encode_plus(list(context), padding=True, return_tensors='pt', max_length=2560,
                                                truncation=True).input_ids.to(device)
        outputs = model.generate(input_ids, num_beams=3, max_length=500, early_stopping=True, no_repeat_ngram_size=3)
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
            'f1': AverageMeter('F1_U', ':6.4f'),
            'bleu': AverageMeter('SacreBleu', ':6.4f'),
            'rouge': AverageMeter('Rouge-L', ':6.4f')
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

        # SacreBleu
        bleu_score = [
            sacrebleu.sentence_bleu(hypothesis, [reference]).score
            for hypothesis, reference in zip(hypothesis_list, reference_list)
        ]
        bleu_score = sum(bleu_score) / instance_num
        meters['bleu'].update(bleu_score, instance_num)

        # Rouge-L
        rouge_func = Rouge()
        rouge_score = [x['rouge-l']['f'] for x in rouge_func.get_scores(hypothesis_list, reference_list)]
        rouge_score = (sum(rouge_score) / instance_num) * 100
        meters['rouge'].update(rouge_score, instance_num)


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
