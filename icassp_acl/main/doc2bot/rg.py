import sys
import copy
import json
import sacrebleu
from rouge import Rouge

from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    BertTokenizer,
    BartForConditionalGeneration,
    RagTokenForGeneration,
    BertForSequenceClassification,
    DPRConfig,
    DPRQuestionEncoder
)

import utils
from utils import AverageMeter
from abstracts import BaseRunner


class Rerank(nn.Module):

    def __init__(self, encoder, passages):
        super().__init__()
        self.encoder = encoder
        self.passages = passages

    def forward(self, inputs):
        model = self.encoder
        logits = F.log_softmax(model(**inputs)[0], dim=-1)[:, 1]  # log_softmax over the binary classification
        logits = logits.view(-1, self.passages)
        logprobs = F.log_softmax(logits, dim=-1)  # log_softmax over the passages
        return None, logprobs


class Re2GModel(nn.Module):
    def __init__(self, rerank, generator):
        super(Re2GModel, self).__init__()
        self.rerank = rerank
        self.generator = generator

    def forward(self, n_docs, rerank_inputs, generator_input_ids, generator_attention_mask, labels):
        _, doc_scores = self.rerank(rerank_inputs)

        outputs = self.generator(
            labels=labels,
            context_input_ids=generator_input_ids,
            context_attention_mask=generator_attention_mask,
            doc_scores=doc_scores,
            n_docs=n_docs
        )
        loss = outputs.loss.mean()
        return loss, outputs

    def generate(self, n_docs, rerank_inputs, generator_input_ids, generator_attention_mask, max_length=500):
        _, doc_scores = self.rerank(rerank_inputs)

        beam_search_output = self.generator.generate(
            n_docs=n_docs,
            encoder_input_ids=generator_input_ids,
            context_input_ids=generator_input_ids,
            context_attention_mask=generator_attention_mask,
            doc_scores=doc_scores,
            num_beams=3,
            max_length=max_length,
            early_stopping=True,
            no_repeat_ngram_size=3,
            return_dict_in_generate=True,
            output_scores=True
        )
        # BeamSearchDecoderOnlyOutput: sequences, sequences_scores
        generated_ids = beam_search_output.sequences.detach().cpu().numpy()
        generated_scores = beam_search_output.sequences_scores.detach().cpu().numpy()

        return generated_ids, generated_scores


class SFRe2GDataset(Dataset):
    def __init__(self, id_list, rerank_list, context_list, query_list, label_list):
        self.id_list = id_list
        self.rerank_list = rerank_list
        self.context_list = context_list
        self.query_list = query_list
        self.label_list = label_list

    def __len__(self):
        return len(self.context_list)

    def __getitem__(self, index):
        ids = self.id_list[index]
        rerank = self.rerank_list[index]
        context = self.context_list[index]
        query = self.query_list[index]
        label = self.label_list[index]
        return ids, rerank, context, query, label


def text2line(text):
    return text.replace("\n", " ").replace("\r", " ").strip()


def btag(tag, text):  # tag the content
    return "<{}>{}</{}>".format(tag, text2line(text), tag)


class SFRe2GRunner(BaseRunner):
    def __init__(self, args):
        super(SFRe2GRunner, self).__init__()
        self.args = args
        if not args.unit_test:
            # prepare rerank
            rerank_path = args.rerank_path if args.rerank_path else 'hfl/chinese-roberta-wwm-ext-large'
            config, tokenizer, encoder = utils.load_pretrain(
                args, AutoConfig, BertTokenizer, BertForSequenceClassification, rerank_path)
            rerank = Rerank(encoder, args.passages)
            self.rerank_tokenizer = tokenizer
            config, tokenizer, generator = utils.load_pretrain(
                args, AutoConfig, BertTokenizer, BartForConditionalGeneration, 'fnlp/bart-large-chinese')
            special_tokens = ["<最后一轮>", "<用户说>", "<系统说>"]
            tokenizer.add_tokens(special_tokens)
            generator.resize_token_embeddings(len(tokenizer))
            config = DPRConfig()
            config.vocab_size = encoder.config.vocab_size
            rag_model = RagTokenForGeneration(question_encoder=DPRQuestionEncoder(config), generator=generator)
            rag_model.rag.question_encoder = None
            model = Re2GModel(rerank, rag_model)
            self.tokenizer = tokenizer
            self.model = model

    def prepare_dataset(self, args, split='train'):
        id_list, rerank_list, context_list, query_list, label_list = [], [], [], [], []

        if split == 'valid':
            split = 'test'

        act_list = ['question/open', 'ans/open', 'multiple-choice', 'ans/yes', 'ans/yesno', 'verification', 'ans/no']

        with open(f'{args.home}/acl/{split}.json', 'r') as f:
            act_data = json.load(f)
            act_data = {x['id']: act_list.index(x['act']) for x in act_data}

        with open(f'{args.home}/acl/{split}_add_origin_passage.jsonl', 'r') as f:
            input_data = f.readlines()

        with open(f'{args.home}/acl/doc2bot-{split}-kilt.jsonl', 'r') as f:
            target_data = f.readlines()

        for inputs, target in zip(input_data, target_data):
            inputs = json.loads(inputs)
            target = json.loads(target)
            assert inputs['id'] == target['id']
            id_list.append(inputs['id'])
            query_ids = self.tokenizer([inputs['input']], add_special_tokens=False, return_tensors='pt')['input_ids'][
                            0][:195]
            query = self.tokenizer.decode(query_ids)
            query_list.append(query)
            rerank_list.append([x['text'] for x in inputs['passages']])
            context_list.append([x['text'] for x in inputs['origin_passages']])
            label_list.append(f"{act_list[act_data[inputs['id']]]} " + target['output'][0]['answer'])

        # create dataset
        dataset = SFRe2GDataset(id_list, rerank_list, context_list, query_list, label_list)
        return dataset

    @staticmethod
    def collate(batch):
        ids = [item[0] for item in batch]
        rerank = [item[1] for item in batch]
        context = [item[2] for item in batch]
        query = [item[3] for item in batch]
        label = [item[4] for item in batch]
        return ids, rerank, context, query, label

    def forward(self, model, payload, tokenizer, device):
        ids, rerank, context, query, label = payload
        if self.args.subgraph:
            context = rerank

        n_docs = self.args.passages
        assert n_docs <= len(context[0])

        querys = [x for x in query for i in range(n_docs)]
        contexts = [x for ctxs in rerank for x in ctxs[:n_docs]]
        assert len(querys) == len(contexts)

        rerank_inputs = self.rerank_tokenizer(
            querys, contexts,
            add_special_tokens=True,
            return_tensors='pt',
            max_length=self.args.source_sequence_size,
            padding='longest',
            truncation=True
        )
        rerank_inputs = {n: t.to(device) for n, t in rerank_inputs.items()}

        # generator inputs
        generator_inputs = [
            query[i] + '[SEP]' + doc
            for i in range(len(query)) for doc in context[i][:n_docs]
        ]
        generator_tokenized_inputs = tokenizer.batch_encode_plus(
            list(generator_inputs), padding=True, return_tensors='pt',
            max_length=self.args.source_sequence_size, truncation=True
        )

        # labels
        label_ids = tokenizer.batch_encode_plus(
            list(label), padding=True, return_tensors='pt', max_length=self.args.target_sequence_size,
            truncation=True
        ).input_ids.to(device)

        outputs = model(
            n_docs=n_docs,
            rerank_inputs=rerank_inputs,
            generator_input_ids=generator_tokenized_inputs.input_ids.to(device),
            generator_attention_mask=generator_tokenized_inputs.attention_mask.to(device),
            labels=label_ids
        )
        return outputs

    def inference(self, model, payload, tokenizer, device, result_dict):
        ids, rerank, context, query, label = payload
        if self.args.subgraph:
            context = rerank

        n_docs = self.args.passages
        assert n_docs <= len(context[0])
        for i in range(len(ids)):
            querys = [query[i] for doc in context[i][:n_docs]]
            contexts = rerank[i][:n_docs]
            assert len(querys) == len(contexts)

            rerank_inputs = self.rerank_tokenizer(
                querys, contexts,
                add_special_tokens=True,
                return_tensors='pt',
                max_length=self.args.source_sequence_size,
                padding='longest',
                truncation=True
            )
            rerank_inputs = {n: t.to(device) for n, t in rerank_inputs.items()}

            # generator inputs
            generator_inputs = [query[i] + '[SEP]' + doc
                                for doc in context[i][:n_docs]]
            generator_tokenized_inputs = tokenizer.batch_encode_plus(
                list(generator_inputs), padding=True, return_tensors='pt',
                max_length=self.args.source_sequence_size, truncation=True
            )

            generated_ids, generated_scores = model.generate(
                n_docs=n_docs,
                rerank_inputs=rerank_inputs,
                generator_input_ids=generator_tokenized_inputs.input_ids.to(device),
                generator_attention_mask=generator_tokenized_inputs.attention_mask.to(device),
                max_length=self.args.target_sequence_size
            )

            hypothesis = tokenizer.batch_decode(generated_ids, skip_special_tokens=True,
                                                clean_up_tokenization_spaces=False)
            result_dict['id_list'].append(ids[i])
            result_dict['outputs'].append(hypothesis[0])
            result_dict['targets'].append(label[i])

    def init_losses_dict(self):
        losses = {
            'loss': AverageMeter('Loss', ':.4e'),
        }
        return copy.deepcopy(losses)

    def measure_loss(self, payload, model_output, losses):
        ids, rerank, context, query, label = payload
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
