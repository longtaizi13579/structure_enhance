import copy
import json
import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    BertConfig,
    BertTokenizer,
    BertForSequenceClassification,
    ElectraConfig,
    ElectraTokenizer,
    ElectraForSequenceClassification,
)

import utils
from utils import AverageMeter
from abstracts import BaseRunner


class PolicyDataset(Dataset):
    def __init__(self, id_list, query_list, passages_list, labels_list):
        self.id_list = id_list
        self.query_list = query_list
        self.passages_list = passages_list
        self.labels_list = labels_list

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, index):
        id_ = self.id_list[index]
        query = self.query_list[index]
        passages = self.passages_list[index]
        label = self.labels_list[index]
        return id_, query, passages, label


class PolicyRunner(BaseRunner):

    def __init__(self, args):
        super(PolicyRunner, self).__init__()
        self.args = args
        if not args.unit_test:
            if args.model == 'bert':
                config, tokenizer, model = utils.load_pretrain(
                    args, BertConfig, BertTokenizer, BertForSequenceClassification, 'hfl/chinese-bert-wwm-ext')
            elif args.model == 'roberta':
                config, tokenizer, model = utils.load_pretrain(
                    args, BertConfig, BertTokenizer, BertForSequenceClassification, 'hfl/chinese-roberta-wwm-ext')
            elif args.model == 'electra':
                config, tokenizer, model = utils.load_pretrain(
                    args, ElectraConfig, ElectraTokenizer, ElectraForSequenceClassification,
                    'hfl/chinese-electra-base-discriminator')
            else:
                raise Exception()

            self.tokenizer = tokenizer
            self.model = model

    def prepare_dataset(self, args, split='train'):
        id_list, query_list, passages_list, pids_list, labels_list = [], [], [], [], []
        if split == 'valid':
            split = 'test'

        passage_map = {}
        with open(f'{args.home}/acl/doc2bot_passages.jsonl', 'r') as f:
            for line in f.readlines():
                sample = json.loads(line)
                passage_map[sample['pid']] = sample['text']
                if args.rm_struct:
                    passage_map[sample['pid']] = sample['text'].split('[SEP]')[0]

        with open(f'{args.home}/acl/doc2bot-{split}-kilt.jsonl', 'r') as f:
            dial_data = [json.loads(line) for line in f.readlines()]

        with open(f'{args.home}/acl/retrieved-{split}.jsonl', 'r') as f:
            retrieved_data = [json.loads(line) for line in f.readlines()]

        for dial, retrieve in zip(dial_data, retrieved_data):
            query = dial['input']
            query_ids = self.tokenizer([query], add_special_tokens=False, return_tensors='pt')['input_ids'][
                            0][:128]
            query = self.tokenizer.decode(query_ids)
            all_passages = [x['wikipedia_id'] for x in retrieve['output'][0]['provenance']]
            positive_passages = [x['wikipedia_id'] for x in dial['output'][0]['provenance']]
            negative_passages = [x for x in all_passages if x not in positive_passages]

            for positive_id in positive_passages:
                id_list.append(dial['id'])
                query_list.append(query)
                passages_list.append(passage_map[positive_id])
                labels_list.append(1)

            for negative_id in negative_passages:
                id_list.append(dial['id'])
                query_list.append(query)
                passages_list.append(passage_map[negative_id])
                labels_list.append(0)

        # create dataset
        dataset = PolicyDataset(id_list, query_list, passages_list, labels_list)
        return dataset

    @staticmethod
    def collate(batch):
        ids = [item[0] for item in batch]
        querys = [item[1] for item in batch]
        passages = [item[2] for item in batch]
        labels = torch.tensor([item[3] for item in batch], dtype=torch.long)
        return ids, querys, passages, labels

    def forward(self, model, payload, tokenizer, device):
        ids, querys, passages, labels = payload
        tokenizer_outputs = tokenizer.batch_encode_plus(zip(querys, passages), padding=True, return_tensors='pt',
                                                        max_length=self.args.source_sequence_size, truncation=True)
        input_ids = tokenizer_outputs.input_ids.to(device)
        attention_mask = tokenizer_outputs.attention_mask.to(device)
        labels = labels.to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs

    def inference(self, model, payload, tokenizer, device, result_dict):
        ids, querys, passages, labels = payload
        tokenizer_outputs = tokenizer.batch_encode_plus(zip(querys, passages), padding=True, return_tensors='pt',
                                                        max_length=self.args.source_sequence_size, truncation=True)
        input_ids = tokenizer_outputs.input_ids.to(device)
        attention_mask = tokenizer_outputs.attention_mask.to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        outputs = torch.argmax(logits, dim=1).tolist()

        for i in range(len(outputs)):
            sample = result_dict.get(ids[i], {
                'outputs': [],
                'targets': []
            })
            if (outputs[i] == 1) and (passages[i] not in sample['outputs']):
                sample['outputs'].append(passages[i])
            if (labels[i].item() == 1) and (passages[i] not in sample['targets']):
                sample['targets'].append(passages[i])
            result_dict[ids[i]] = sample

    def init_losses_dict(self):
        losses = {
            'loss': AverageMeter('Loss', ':.4e'),
        }
        return copy.deepcopy(losses)

    def measure_loss(self, payload, model_output, losses):
        ids, querys, passages, labels = payload
        loss = model_output[0]
        losses['loss'].update(loss.item(), len(ids))

    def init_results_dict(self):
        results = {
        }
        return copy.deepcopy(results)

    def init_meters_dict(self):
        meters = {
            'F1': AverageMeter('F1', ':6.4f')
        }
        return copy.deepcopy(meters)

    def measure_result(self, result_dict, meters):
        for k, v in result_dict.items():
            predict_relevant_set = set(v['outputs'])
            ground_set = set(v['targets'])
            tp = len(predict_relevant_set.intersection(ground_set))
            tp_fp = len(predict_relevant_set)
            tp_fn = len(ground_set)
            f1 = 0
            if (tp_fn != 0) and (tp_fp != 0) and (tp != 0):
                p = tp / tp_fp
                r = tp / tp_fn
                f1 = ((2 * p * r) / (p + r)) * 100
            meters['F1'].update(f1)
