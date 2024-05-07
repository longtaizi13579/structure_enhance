import os
import sys
import copy
import json
import types

from rouge import Rouge
import sacrebleu

import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer, T5ForConditionalGeneration, T5Config, AutoConfig, BertTokenizer
import torch.nn.functional as F

import utils
from transformers.modeling_outputs import BaseModelOutput
from utils import AverageMeter
from abstracts import BaseRunner


class FiDDataset(Dataset):
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


class FiDRunner(BaseRunner):
    def __init__(self, args):
        super(FiDRunner, self).__init__()
        if args.language == 'en':
            pretrained_path = f'{args.home}/{args.code}/pretrained/{args.model}/'
            if not os.path.exists(pretrained_path):
                utils.mk_dir(pretrained_path)
                tokenizer = AutoTokenizer.from_pretrained(args.model)
                special_tokens = ["<last_turn>", "<user>", "<agent>", "<response>", "<grounding>"]
                tokenizer.add_tokens(special_tokens)
                tokenizer.save_pretrained(pretrained_path)
                config = T5Config.from_pretrained(args.model)
                config.dropout_rate = args.dropout
                t5 = T5ForConditionalGeneration.from_pretrained(args.model)
                t5.resize_token_embeddings(len(tokenizer))
                model = FiDT5(t5.config)
                model.load_t5(t5.state_dict())
                config.save_pretrained(pretrained_path)
                model.save_pretrained(pretrained_path)
            else:
                tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
                config = T5Config.from_pretrained(pretrained_path)
                config.dropout_rate = args.dropout
                model = FiDT5.from_pretrained(pretrained_path, config=config)
        else:
            config, tokenizer, t5 = utils.load_pretrain(
                args, AutoConfig, AutoTokenizer, T5ForConditionalGeneration, args.model)
            special_tokens = ["<最后一轮>", "<用户说>", "<系统说>"]
            tokenizer.add_tokens(special_tokens)
            t5.resize_token_embeddings(len(tokenizer))
            model = FiDT5(t5.config)
            model.load_t5(t5.state_dict())
        self.args = args
        self.tokenizer = tokenizer
        self.model = model

    def prepare_dataset(self, args, split='train'):
        id_list, context_list, label_list = [], [], []
        if args.language == 'en':
            if split == 'valid':
                split = 'validation'

            with open(f'{args.home}/g4/multidoc2dial/candidates.json', 'r') as f:
                candidates = json.load(f)

            docs_map = {}
            with open(f'{args.home}/g4/multidoc2dial/{split}.jsonl', 'r') as f:
                for line in f.readlines():
                    sample = json.loads(line)
                    docs_map[sample['id']] = [
                        candidates[int(x['wikipedia_id'])] for x in sample['output'][0]['provenance'][:5]
                    ]
            with open(f'{args.home}/g4/multidoc2dial/{split}-kilt.jsonl', 'r') as f:
                for line in f.readlines():
                    sample = json.loads(line)
                    id_list.append(sample['id'])
                    query_ids = \
                        self.tokenizer([sample['input']], add_special_tokens=False, return_tensors='pt')['input_ids'][
                            0][
                        :195]
                    query = self.tokenizer.decode(query_ids)
                    context_list.append([
                        query + '[SEP]' + passage for passage in docs_map[sample['id']]
                    ])
                    label_list.append(sample['output'][0]['answer'])
        else:
            if split == 'valid':
                split = 'test'

            with open(f'{args.home}/g4/candidates.json', 'r') as f:
                candidates = json.load(f)

            docs_map = {}
            with open(f'{args.home}/g4/{split}.jsonl', 'r') as f:
                for line in f.readlines():
                    sample = json.loads(line)
                    docs_map[sample['id']] = [
                        candidates[int(x['wikipedia_id'])] for x in sample['output'][0]['provenance'][:5]
                    ]
            with open(f'{args.home}/g4/doc2bot-{split}-kilt.jsonl', 'r') as f:
                for line in f.readlines():
                    sample = json.loads(line)
                    id_list.append(sample['id'])
                    query_ids = \
                        self.tokenizer([sample['input']], add_special_tokens=False, return_tensors='pt')['input_ids'][
                            0][
                        :195]
                    query = self.tokenizer.decode(query_ids)
                    context_list.append([
                        query + '[SEP]' + passage for passage in docs_map[sample['id']]
                    ])
                    label_list.append(sample['output'][0]['answer'])

        # create dataset
        dataset = FiDDataset(id_list, context_list, label_list)
        return dataset

    def collate(self, batch):
        id_list = [item[0] for item in batch]
        context_list = [item[1] for item in batch]
        label_list = [item[2] for item in batch]
        return id_list, context_list, label_list

    def forward(self, model, payload, tokenizer, device):
        ids, contexts, label = payload
        input_ids, attention_mask = [], []
        for context in contexts:
            tokenizer_outputs = tokenizer.batch_encode_plus(list(context), padding='max_length', return_tensors='pt',
                                                            max_length=self.args.source_sequence_size, truncation=True)
            final_ids = tokenizer_outputs['input_ids'][None]
            final_masks = tokenizer_outputs['attention_mask'][None]
            input_ids.append(final_ids)
            attention_mask.append(final_masks)
        input_ids = torch.cat(input_ids, dim=0).to(device)
        attention_mask = torch.cat(attention_mask, dim=0).bool().to(device)
        label_ids = tokenizer.batch_encode_plus(
            list(label), padding='max_length', return_tensors='pt', max_length=self.args.target_sequence_size,
            truncation=True).input_ids.to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=label_ids)
        return outputs

    def inference(self, model, payload, tokenizer, device, result_dict):
        ids, contexts, label = payload
        input_ids, attention_mask = [], []
        for context in contexts:
            tokenizer_outputs = tokenizer.batch_encode_plus(list(context), padding='max_length', return_tensors='pt',
                                                            max_length=self.args.source_sequence_size, truncation=True)
            final_ids = tokenizer_outputs['input_ids'][None]
            final_masks = tokenizer_outputs['attention_mask'][None]
            input_ids.append(final_ids)
            attention_mask.append(final_masks)
        input_ids = torch.cat(input_ids, dim=0).to(device)
        attention_mask = torch.cat(attention_mask, dim=0).bool().to(device)
        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                 max_length=self.args.target_sequence_size, num_beams=self.args.beam_size)
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

        if self.args.language == 'en':
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
        else:
            sys.setrecursionlimit(512 * 2048)
            hypothesis_list = [x.replace(' ', '').split('<系统说>：')[-1].strip() for x in result_dict['outputs']]
            hypothesis_list = [x if x else '@' for x in hypothesis_list]
            reference_list = [x.split('<系统说>：')[-1].strip() for x in result_dict['targets']]
            hypothesis_list = [' '.join(x) for x in hypothesis_list]
            reference_list = [' '.join(x) for x in reference_list]
            instance_num = len(hypothesis_list)

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


class FiDT5(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.wrap_encoder()

    def forward_(self, **kwargs):
        if 'input_ids' in kwargs:
            kwargs['input_ids'] = kwargs['input_ids'].view(kwargs['input_ids'].size(0), -1)
        if 'attention_mask' in kwargs:
            kwargs['attention_mask'] = kwargs['attention_mask'].view(kwargs['attention_mask'].size(0), -1)

        return super(FiDT5, self).forward(
            **kwargs
        )

    # We need to resize as B x (N * L) instead of (B * N) x L here
    # because the T5 forward method uses the input tensors to infer
    # dimensions used in the decoder.
    # EncoderWrapper resizes the inputs as (B * N) x L.
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        if input_ids != None:
            # inputs might have already be resized in the generate method
            if input_ids.dim() == 3:
                self.encoder.n_passages = input_ids.size(1)
            input_ids = input_ids.view(input_ids.size(0), -1)
        if attention_mask != None:
            attention_mask = attention_mask.view(attention_mask.size(0), -1)
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

    # We need to resize the inputs here, as the generate method expect 2D tensors
    def generate(self, input_ids, attention_mask, max_length, num_beams):
        self.encoder.n_passages = input_ids.size(1)
        return super().generate(
            input_ids=input_ids.view(input_ids.size(0), -1),
            attention_mask=attention_mask.view(attention_mask.size(0), -1),
            max_length=max_length,
            num_beams=num_beams
        )

    def wrap_encoder(self, use_checkpoint=False):
        """
        Wrap T5 encoder to obtain a Fusion-in-Decoder model.
        """
        self.encoder = EncoderWrapper(self.encoder, use_checkpoint=use_checkpoint)

    def unwrap_encoder(self):
        """
        Unwrap Fusion-in-Decoder encoder, useful to load T5 weights.
        """
        self.encoder = self.encoder.encoder
        block = []
        for mod in self.encoder.block:
            block.append(mod.module)
        block = nn.ModuleList(block)
        self.encoder.block = block

    def load_t5(self, state_dict):
        self.unwrap_encoder()
        self.load_state_dict(state_dict)
        self.wrap_encoder()

    def set_checkpoint(self, use_checkpoint):
        """
        Enable or disable checkpointing in the encoder.
        See https://pytorch.org/docs/stable/checkpoint.html
        """
        for mod in self.encoder.encoder.block:
            mod.use_checkpoint = use_checkpoint

    def reset_score_storage(self):
        """
        Reset score storage, only used when cross-attention scores are saved
        to train a retriever.
        """
        for mod in self.decoder.block:
            mod.layer[1].EncDecAttention.score_storage = None

    def get_crossattention_scores(self, context_mask):
        """
        Cross-attention scores are aggregated to obtain a single scalar per
        passage. This scalar can be seen as a similarity score between the
        question and the input passage. It is obtained by averaging the
        cross-attention scores obtained on the first decoded token over heads,
        layers, and tokens of the input passage.

        More details in Distilling Knowledge from Reader to Retriever:
        https://arxiv.org/abs/2012.04584.
        """
        scores = []
        n_passages = context_mask.size(1)
        for mod in self.decoder.block:
            scores.append(mod.layer[1].EncDecAttention.score_storage)
        scores = torch.cat(scores, dim=2)
        bsz, n_heads, n_layers, _ = scores.size()
        # batch_size, n_head, n_layers, n_passages, text_maxlength
        scores = scores.view(bsz, n_heads, n_layers, n_passages, -1)
        scores = scores.masked_fill(~context_mask[:, None, None], 0.)
        scores = scores.sum(dim=[1, 2, 4])
        ntokens = context_mask.sum(dim=[2]) * n_layers * n_heads
        scores = scores / ntokens
        return scores

    def overwrite_forward_crossattention(self):
        """
        Replace cross-attention forward function, only used to save
        cross-attention scores.
        """
        for mod in self.decoder.block:
            attn = mod.layer[1].EncDecAttention
            attn.forward = types.MethodType(cross_attention_forward, attn)


class EncoderWrapper(torch.nn.Module):
    """
    Encoder Wrapper for T5 Wrapper to obtain a Fusion-in-Decoder model.
    """

    def __init__(self, encoder, use_checkpoint=False):
        super().__init__()

        self.encoder = encoder
        apply_checkpoint_wrapper(self.encoder, use_checkpoint)

    def forward(self, input_ids=None, attention_mask=None, **kwargs, ):
        # total_length = n_passages * passage_length
        bsz, total_length = input_ids.shape
        passage_length = total_length // self.n_passages
        input_ids = input_ids.view(bsz * self.n_passages, passage_length)
        attention_mask = attention_mask.view(bsz * self.n_passages, passage_length)
        outputs = self.encoder(input_ids, attention_mask, **kwargs)
        encoder_outputs = (outputs[0].view(bsz, self.n_passages * passage_length, -1),) + outputs[1:]
        return BaseModelOutput(
            last_hidden_state=encoder_outputs[0],
            hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
            attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
        )


class CheckpointWrapper(torch.nn.Module):
    """
    Wrapper replacing None outputs by empty tensors, which allows the use of
    checkpointing.
    """

    def __init__(self, module, use_checkpoint=False):
        super().__init__()
        self.module = module
        self.use_checkpoint = use_checkpoint

    def forward(self, hidden_states, attention_mask, position_bias, **kwargs):
        if self.use_checkpoint and self.training:
            kwargs = {k: v for k, v in kwargs.items() if v is not None}

            def custom_forward(*inputs):
                output = self.module(*inputs, **kwargs)
                empty = torch.tensor(
                    [],
                    dtype=torch.float,
                    device=output[0].device,
                    requires_grad=True)
                output = tuple(x if x is not None else empty for x in output)
                return output

            output = torch.utils.checkpoint.checkpoint(
                custom_forward,
                hidden_states,
                attention_mask,
                position_bias
            )
            output = tuple(x if x.size() != 0 else None for x in output)
        else:
            output = self.module(hidden_states, attention_mask, position_bias, **kwargs)
        return output


def apply_checkpoint_wrapper(t5stack, use_checkpoint):
    """
    Wrap each block of the encoder to enable checkpointing.
    """
    block = []
    for mod in t5stack.block:
        wrapped_mod = CheckpointWrapper(mod, use_checkpoint)
        block.append(wrapped_mod)
    block = nn.ModuleList(block)
    t5stack.block = block


def cross_attention_forward(
        self,
        input,
        mask=None,
        kv=None,
        position_bias=None,
        past_key_value_state=None,
        head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
):
    """
    This only works for computing cross attention over the input
    """
    assert (kv != None)
    assert (head_mask == None)
    assert (position_bias != None or self.has_relative_attention_bias)

    bsz, qlen, dim = input.size()
    n_heads, d_heads = self.n_heads, self.d_kv
    klen = kv.size(1)

    q = self.q(input).view(bsz, -1, n_heads, d_heads).transpose(1, 2)
    if past_key_value_state == None:
        k = self.k(kv).view(bsz, -1, n_heads, d_heads).transpose(1, 2)
        v = self.v(kv).view(bsz, -1, n_heads, d_heads).transpose(1, 2)
    else:
        k, v = past_key_value_state

    scores = torch.einsum("bnqd,bnkd->bnqk", q, k)

    if mask is not None:
        scores += mask

    if position_bias is None:
        position_bias = self.compute_bias(qlen, klen)
    scores += position_bias

    if self.score_storage is None:
        self.score_storage = scores

    attn = F.softmax(scores.float(), dim=-1).type_as(scores)
    attn = F.dropout(attn, p=self.dropout, training=self.training)

    output = torch.matmul(attn, v)
    output = output.transpose(1, 2).contiguous().view(bsz, -1, self.inner_dim)
    output = self.o(output)

    if use_cache:
        output = (output,) + ((k, v),)
    else:
        output = (output,) + (None,)

    if output_attentions:
        output = output + (attn,)

    if self.has_relative_attention_bias:
        output = output + (position_bias,)

    return output
