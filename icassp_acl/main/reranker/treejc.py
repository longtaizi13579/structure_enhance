import copy
import json
import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import RobertaConfig, RobertaTokenizer, RobertaModel

import utils
from utils import AverageMeter
from abstracts import BaseRunner


class Model(nn.Module):
    def __init__(self, roberta, passages=20, dropout=0.1):
        super(Model, self).__init__()
        self.roberta = roberta
        self.passages = passages
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(roberta.config.hidden_size, 1)
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        logits = logits.view(-1, self.passages)

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels)

        return loss, logits


class TreeJCDataset(Dataset):
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


class TreeJCRunner(BaseRunner):

    def __init__(self, args):
        super(TreeJCRunner, self).__init__()
        self.args = args
        if not args.unit_test:
            config, tokenizer, longformer = utils.load_pretrain(args, RobertaConfig, RobertaTokenizer,
                                                                RobertaModel)
            model = Model(longformer, args.passages, args.dropout)
            self.tokenizer = tokenizer
            self.model = model

    def prepare_dataset(self, args, split='train'):
        id_list, query_list, passages_list, pids_list, labels_list = [], [], [], [], []
        data_map = {
            "train": "dpr_train",
            "valid": "dpr_dev"
        }
        with open(f'{args.home}/g4/{data_map[split]}.json', 'r') as f:
            dial_data = json.load(f)

        for dial in dial_data:
            query = ' '.join(' '.join(dial['question']).split(' ')[:128])
            if split == 'train':
                if dial['grd_psg'] in dial['ctxs']:
                    dial['ctxs'].pop(dial['ctxs'].index(dial['grd_psg']))
                passages = [dial['grd_psg']] + dial['ctxs'][:19]
                label = 0
            else:
                passages = dial['ctxs']
                label = -1
                if dial['grd_psg'] in dial['ctxs']:
                    label = dial['ctxs'].index(dial['grd_psg'])
            id_list.append(dial['id'])
            query_list.append(query)
            passages_list.append(passages)
            labels_list.append(label)

        # create dataset
        dataset = TreeJCDataset(id_list, query_list, passages_list, labels_list)
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
        querys = [query for query in querys for i in range(self.args.passages)]
        contexts = [context for contexts in passages for context in contexts]
        tokenizer_outputs = tokenizer.batch_encode_plus(zip(querys, contexts), padding=True, return_tensors='pt',
                                                        max_length=self.args.source_sequence_size, truncation=True)
        input_ids = tokenizer_outputs.input_ids.to(device)
        attention_mask = tokenizer_outputs.attention_mask.to(device)
        labels = labels.to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs

    def inference(self, model, payload, tokenizer, device, result_dict):
        ids, querys, passages, labels = payload
        querys = [query for query in querys for i in range(self.args.passages)]
        contexts = [context for contexts in passages for context in contexts]
        tokenizer_outputs = tokenizer.batch_encode_plus(zip(querys, contexts), padding=True, return_tensors='pt',
                                                        max_length=self.args.source_sequence_size, truncation=True)
        input_ids = tokenizer_outputs.input_ids.to(device)
        attention_mask = tokenizer_outputs.attention_mask.to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask)[1].cpu().numpy()
        sorted_idxs = []
        for logit in logits:
            sorted_idx = sorted(zip(range(len(logit)), logit), key=lambda x: x[1], reverse=True)
            sorted_idxs.append([x[0] for x in sorted_idx])
        result_dict['id_list'] += ids
        result_dict['targets'] += labels.tolist()
        result_dict['outputs'] += [x for x in sorted_idxs]

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
            'id_list': [],
            'outputs': [],
            'targets': []
        }
        return copy.deepcopy(results)

    def init_meters_dict(self):
        meters = {
            'R@1': AverageMeter('R@1', ':6.4f'),
            'R@5': AverageMeter('R@5', ':6.4f')
        }
        return copy.deepcopy(meters)

    def measure_result(self, result_dict, meters):
        outputs = result_dict['outputs']
        targets = result_dict['targets']
        for output, target in zip(outputs, targets):
            r1 = 1 if target == output[0] else 0
            meters['R@1'].update(r1)
            r5 = 1 if target in output[:5] else 0
            meters['R@5'].update(r5)


if __name__ == '__main__':
    from options import Options

    options = Options()
    args = options.parse()
    args.passages = 5
    args.unit_test = True
    runner = TreeJCRunner(args)
    runner.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    runner.model = Model(RobertaModel.from_pretrained('roberta-base'), args.passages)

    ids = ['1', '2']
    querys = ['question 1', 'question 2']
    passages = [['question 1'] * 5, ['question 2'] * 5]
    # pids = [[i for i in range(5)], [i for i in range(5)]]
    labels = torch.tensor([0, 1], dtype=torch.long)
    payload = ids, querys, passages, labels
    outputs = runner.forward(runner.model, payload, runner.tokenizer, 'cpu')

    with torch.no_grad():
        result_dict = runner.init_results_dict()
        runner.inference(runner.model, payload, runner.tokenizer, 'cpu', result_dict)
        print(result_dict)

    meters = runner.init_meters_dict()
    runner.measure_result(result_dict, meters)
    for k, v in meters.items():
        print(f'{k} \t {v}')
