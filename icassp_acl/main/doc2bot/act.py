import copy
import json
import torch
from sklearn.metrics import f1_score
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


class ActDataset(Dataset):
    def __init__(self, id_list, query_list, labels_list):
        self.id_list = id_list
        self.query_list = query_list
        self.labels_list = labels_list

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, index):
        id_ = self.id_list[index]
        query = self.query_list[index]
        label = self.labels_list[index]
        return id_, query, label


act_list = ['question/open', 'ans/open', 'multiple-choice', 'ans/yes', 'ans/yesno', 'verification', 'ans/no']


class ActRunner(BaseRunner):

    def __init__(self, args):
        super(ActRunner, self).__init__()
        self.args = args
        if not args.unit_test:
            args.num_labels = len(act_list)
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

        with open(f'{args.home}/acl/{split}.json', 'r') as f:
            act_data = json.load(f)
            act_data = {x['id']: act_list.index(x['act']) for x in act_data}

        with open(f'{args.home}/acl/doc2bot-{split}-kilt.jsonl', 'r') as f:
            dial_data = [json.loads(line) for line in f.readlines()]

        for dial in dial_data:
            query = dial['input']
            query_ids = self.tokenizer([query], add_special_tokens=False, return_tensors='pt')['input_ids'][
                            0][:128]
            query = self.tokenizer.decode(query_ids)
            label = act_data[dial['id']]
            id_list.append(dial['id'])
            query_list.append(query)
            labels_list.append(label)

        # create dataset
        dataset = ActDataset(id_list, query_list, labels_list)
        return dataset

    @staticmethod
    def collate(batch):
        ids = [item[0] for item in batch]
        querys = [item[1] for item in batch]
        labels = torch.tensor([item[2] for item in batch], dtype=torch.long)
        return ids, querys, labels

    def forward(self, model, payload, tokenizer, device):
        ids, querys, labels = payload
        tokenizer_outputs = tokenizer.batch_encode_plus(querys, padding=True, return_tensors='pt',
                                                        max_length=self.args.source_sequence_size, truncation=True)
        input_ids = tokenizer_outputs.input_ids.to(device)
        attention_mask = tokenizer_outputs.attention_mask.to(device)
        labels = labels.to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs

    def inference(self, model, payload, tokenizer, device, result_dict):
        ids, querys, labels = payload
        tokenizer_outputs = tokenizer.batch_encode_plus(querys, padding=True, return_tensors='pt',
                                                        max_length=self.args.source_sequence_size, truncation=True)
        input_ids = tokenizer_outputs.input_ids.to(device)
        attention_mask = tokenizer_outputs.attention_mask.to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        outputs = torch.argmax(logits, dim=1).tolist()

        for i in range(len(outputs)):
            result_dict[ids[i]] = {
                'outputs': act_list[outputs[i]],
                'targets': act_list[labels[i].item()]
            }

    def init_losses_dict(self):
        losses = {
            'loss': AverageMeter('Loss', ':.4e'),
        }
        return copy.deepcopy(losses)

    def measure_loss(self, payload, model_output, losses):
        ids, querys, labels = payload
        loss = model_output[0]
        losses['loss'].update(loss.item(), len(ids))

    def init_results_dict(self):
        results = {
        }
        return copy.deepcopy(results)

    def init_meters_dict(self):
        meters = {
            'ma-F1': AverageMeter('ma-F1', ':6.4f'),
            'mi-F1': AverageMeter('mi-F1', ':6.4f')
        }
        return copy.deepcopy(meters)

    def measure_result(self, result_dict, meters):
        predicts = []
        grounds = []
        for v in result_dict.values():
            predicts.append(v['outputs'])
            grounds.append(v['targets'])

        meters['ma-F1'].update(f1_score(grounds, predicts, average='macro'))
        meters['mi-F1'].update(f1_score(grounds, predicts, average='micro'))
