import os
import copy
import json

import faiss
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

from transformers import BertTokenizer, BertModel, BertConfig

import utils
from utils import AverageMeter
from abstracts import BaseRunner
from retriever.bm25 import BM25Tokenizer, BM25


class Wrapper(nn.Module):
    def __init__(self, encoder):
        super(Wrapper, self).__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask, dummy_tensor):
        return self.encoder(input_ids, attention_mask).pooler_output


class Model(nn.Module):
    def __init__(self, qry_model, ctx_model):
        super(Model, self).__init__()
        self.qry_model = Wrapper(qry_model)
        self.ctx_model = Wrapper(ctx_model)
        self.loss_fct = nn.CrossEntropyLoss()

    @staticmethod
    def encode(model, input_ids, attention_mask, gck_segment=32):
        dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)
        pooled_output = []
        for mini_batch in range(0, input_ids.shape[0], gck_segment):
            mini_batch_input_ids = input_ids[mini_batch:mini_batch + gck_segment]
            mini_batch_attention_mask = attention_mask[mini_batch:mini_batch + gck_segment]
            mini_batch_pooled_output = checkpoint(model, mini_batch_input_ids, mini_batch_attention_mask, dummy_tensor)
            pooled_output.append(mini_batch_pooled_output)
        return torch.cat(pooled_output, dim=0)

    def forward(self, query_input_ids, query_attention_mask, context_input_ids, context_attention_mask, labels,
                gck_segment=32):
        query_vector = self.encode(self.qry_model, query_input_ids, query_attention_mask, gck_segment)
        context_vector = self.encode(self.ctx_model, context_input_ids, context_attention_mask, gck_segment)
        logits = torch.matmul(query_vector, context_vector.T)
        loss = self.loss_fct(logits, labels)
        return loss, logits


def add_parent(graph):
    for k, v in graph.items():
        result = []
        pid = v['pid']
        while pid in graph:
            result.append(graph[pid]['data'])
            pid = graph[pid]['pid']
        v['parent'] = result
    return graph


def get_child_map(graph):
    child_map = dict()
    for k, v in graph.items():
        result = child_map.get(v['pid'], [])
        result.append(k)
        child_map[v['pid']] = result
    return child_map


def get_text(args, graph, node_id):
    node = graph[node_id]
    text = node['data']
    if args.add_parent:
        text += f' [SEP] {" // ".join([x.strip() for x in node["parent"]])}'
    return text.strip()


class DPRDataset(Dataset):
    def __init__(self, id_list, query_list, positive_list, negative_list):
        self.id_list = id_list
        self.query_list = query_list
        self.positive_list = positive_list
        self.negative_list = negative_list

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, index):
        id_ = self.id_list[index]
        query = self.query_list[index]
        positive = self.positive_list[index]
        negative = self.negative_list[index]
        return id_, query, positive, negative


class DPRRunner(BaseRunner):

    def __init__(self, args):
        super(DPRRunner, self).__init__()
        self.args = args
        if not args.unit_test:
            config, tokenizer, qry_model = utils.load_pretrain(args, BertConfig, BertTokenizer, BertModel, args.model)
            config, tokenizer, ctx_model = utils.load_pretrain(args, BertConfig, BertTokenizer, BertModel, args.model)
            if args.no_pretrain:
                qry_model = BertModel(config)
                ctx_model = BertModel(config)
            model = Model(qry_model, ctx_model)

            domains = ['company', 'health', 'individual', 'insurance', 'technology', 'wikihow']
            candidates = set()
            documents = dict()
            for domain in domains:
                # process document
                for file in os.listdir(f'{args.home}/doc2bot/train/{domain}/documents/'):
                    if '.json' in file:
                        with open(f'{args.home}/doc2bot/train/{domain}/documents/{file}') as f:
                            document = json.load(f)
                            graph = document['graph']
                            if args.add_parent:
                                graph = add_parent(graph)
                            documents[file] = graph
                            for x in graph:
                                candidates.add(get_text(args, graph, x))

                # process navigation
                with open(f'{args.home}/doc2bot/train/{domain}/navigation.json') as f:
                    navigation = json.load(f)
                    graph = navigation['graph']
                    if args.add_parent:
                        graph = add_parent(graph)
                    documents[domain] = graph
                    for x in graph:
                        candidates.add(get_text(args, graph, x))

            self.tokenizer = tokenizer
            self.model = model
            self.candidates = list(candidates)
            self.documents = documents
            self.index = None

    def prepare_dataset(self, args, split='train'):
        id_list, query_list, positive_list, negative_list = [], [], [], []
        domains = ['company', 'health', 'individual', 'insurance', 'technology', 'wikihow']

        # bm25 index
        with open(f'{args.home}/doc2bot/stop_word.json') as f:
            stopwords = json.load(f)
        tokenizer = BM25Tokenizer(stopwords)
        bm25 = BM25(tokenizer, self.candidates)

        for domain in domains:
            # process dialog
            for file in os.listdir(f'{args.home}/doc2bot/{split}/{domain}/dialogs/'):
                if '.json' in file:
                    with open(f'{args.home}/doc2bot/{split}/{domain}/dialogs/{file}') as f:
                        dialog = json.load(f)
                        history = []
                        for turn in dialog:
                            if turn['role'] == 'user':
                                query = f'<最后一轮>：{turn["utterance"]} [SEP] {" ".join(reversed(history))}'
                                history.append(f'<用户说>：{turn["utterance"]}')
                            else:
                                grounding = []
                                for node_id in turn['grounding_id']:
                                    if 'T' in node_id:
                                        node_id = node_id[1:]
                                        graph = self.documents[domain]
                                    else:
                                        graph = self.documents[turn['document']]
                                    grounding.append(get_text(args, graph, node_id))

                                negative = ''
                                for x in bm25.get_top_n(' '.join(turn['grounding']), 10):
                                    if x not in grounding:
                                        negative = x
                                        break
                                for positive in grounding:
                                    id_list.append(f'{file}:{turn["turn"]}')
                                    query_list.append(query)
                                    positive_list.append(positive)
                                    negative_list.append(negative)
                                history.append(f'<系统说>：{turn["utterance"]}')

        # create dataset
        dataset = DPRDataset(id_list, query_list, positive_list, negative_list)
        return dataset

    @staticmethod
    def collate(batch):
        ids = [item[0] for item in batch]
        querys = [item[1] for item in batch]
        positives = [item[2] for item in batch]
        negatives = [item[3] for item in batch]
        return ids, querys, positives, negatives

    def forward(self, model, payload, tokenizer, device):
        ids, querys, positives, negatives = payload

        # query
        tokenizer_outputs = tokenizer.batch_encode_plus(
            querys,
            padding=True, return_tensors='pt', max_length=self.args.source_sequence_size, truncation=True
        )
        query_input_ids = tokenizer_outputs.input_ids.to(device)
        query_attention_mask = tokenizer_outputs.attention_mask.to(device)

        # context
        tokenizer_outputs = tokenizer.batch_encode_plus(
            positives + negatives,
            padding=True, return_tensors='pt', max_length=self.args.source_sequence_size, truncation=True
        )
        context_input_ids = tokenizer_outputs.input_ids.to(device)
        context_attention_mask = tokenizer_outputs.attention_mask.to(device)

        # label
        labels = torch.tensor(list(range(len(querys))), dtype=torch.long).to(device)

        outputs = model(query_input_ids, query_attention_mask, context_input_ids, context_attention_mask, labels,
                        self.args.gradient_checkpoint_segments)
        return outputs

    def before_inference(self, model, tokenizer, device, result_dict):
        all_ctx_vector = []
        for mini_batch in tqdm(range(0, len(self.candidates), self.args.eval_batch_size)):
            contexts = self.candidates[mini_batch:mini_batch + self.args.eval_batch_size]
            tokenizer_outputs = tokenizer.batch_encode_plus(
                contexts,
                padding=True, return_tensors='pt', max_length=self.args.source_sequence_size, truncation=True
            )
            context_input_ids = tokenizer_outputs.input_ids.to(device)
            context_attention_mask = tokenizer_outputs.attention_mask.to(device)
            sub_ctx_vector = model.ctx_model(context_input_ids, context_attention_mask, None).cpu().numpy()
            all_ctx_vector.append(sub_ctx_vector)

        all_ctx_vector = np.concatenate(all_ctx_vector, axis=0)
        all_ctx_vector = np.array(all_ctx_vector).astype('float32')
        index = faiss.IndexFlatIP(all_ctx_vector.shape[-1])
        index.add(all_ctx_vector)
        self.index = index

    def inference(self, model, payload, tokenizer, device, result_dict):
        ids, querys, positives, negatives = payload

        tokenizer_outputs = tokenizer.batch_encode_plus(querys, padding=True, return_tensors='pt',
                                                        max_length=self.args.source_sequence_size, truncation=True)
        input_ids = tokenizer_outputs.input_ids.to(device)
        attention_mask = tokenizer_outputs.attention_mask.to(device)
        query_vector = model.qry_model(input_ids, attention_mask, None).cpu().numpy().astype('float32')

        D, I = self.index.search(query_vector, 100)
        result_dict['id_list'] += ids
        result_dict['targets'] += [self.candidates.index(x) for x in positives]
        result_dict['outputs'] += I.tolist()

    @staticmethod
    def init_losses_dict():
        losses = {
            'loss': AverageMeter('Loss', ':.4e'),
        }
        return copy.deepcopy(losses)

    @staticmethod
    def measure_loss(payload, model_output, losses):
        ids, querys, positives, negatives = payload
        loss = model_output[0]
        losses['loss'].update(loss.item(), len(ids))

    @staticmethod
    def init_results_dict():
        results = {
            'id_list': [],
            'outputs': [],
            'targets': []
        }
        return copy.deepcopy(results)

    @staticmethod
    def init_meters_dict():
        meters = {
            'R@1': AverageMeter('R@1', ':6.4f'),
            'R@5': AverageMeter('R@5', ':6.4f'),
            'R@100': AverageMeter('R@100', ':6.4f'),
            'MRR@5': AverageMeter('MRR@5', ':6.4f'),
        }
        return copy.deepcopy(meters)

    @staticmethod
    def measure_result(result_dict, meters):
        outputs = result_dict['outputs']
        targets = result_dict['targets']
        for output, target in zip(outputs, targets):
            r1 = 1 if target == output[0] else 0
            meters['R@1'].update(r1)
            r5 = 1 if target in output[:5] else 0
            meters['R@5'].update(r5)
            r100 = 1 if target in output else 0
            meters['R@100'].update(r100)
            if target in output[:5]:
                meters['MRR@5'].update(1 / (output.index(target) + 1))
            else:
                meters['MRR@5'].update(0)


if __name__ == '__main__':
    # with open('../data/result-test.json') as f:
    #     result_dict = json.load(f)
    #
    # meters = DPRRunner.init_meters_dict()
    # DPRRunner.measure_result(result_dict, meters)
    # print(meters)

    # from options import Options
    #
    # options = Options()
    # args = options.parse()
    # args.passages = 5
    # args.unit_test = True
    # runner = DPRRunner(args)
    # runner.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    # runner.model = Model(RobertaModel.from_pretrained('roberta-base'), args.passages)
    #
    # ids = ['1', '2']
    # querys = ['question 1', 'question 2']
    # passages = [['question 1'] * 5, ['question 2'] * 5]
    # # pids = [[i for i in range(5)], [i for i in range(5)]]
    # labels = torch.tensor([0, 1], dtype=torch.long)
    # payload = ids, querys, passages, labels
    # outputs = runner.forward(runner.model, payload, runner.tokenizer, 'cpu')
    #
    # with torch.no_grad():
    #     result_dict = runner.init_results_dict()
    #     runner.inference(runner.model, payload, runner.tokenizer, 'cpu', result_dict)
    #     print(result_dict)
    #
    # meters = runner.init_meters_dict()
    # runner.measure_result(result_dict, meters)
    # for k, v in meters.items():
    #     print(f'{k} \t {v}')
    pass
