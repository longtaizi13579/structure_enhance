import os
import copy

import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel, AutoConfig, BertModel
from torch_geometric.nn import RGCNConv
import utils
from utils import AverageMeter
from abstracts import BaseRunner


class StructureRerankModel(nn.Module):

    def __init__(self, pretrain_model: BertModel, dropout=0.1, tau=0.2):
        super(StructureRerankModel, self).__init__()
        self.pretrain_model = pretrain_model
        self.conv1 = RGCNConv(768, 768, 2)
        self.conv2 = RGCNConv(768, 768, 2)
        self.classifier = nn.Sequential(
            nn.Linear(768, 1),
            nn.Sigmoid()
        )
        self.tau = tau
        self.dropout = nn.Dropout(dropout)

    def forward(self, document_title_input_ids, passages_input_ids, edge_index, edge_type, labels=None):
        document_title_feature = self.pretrain_model(input_ids=document_title_input_ids).last_hidden_state
        document_title_feature = self.dropout(document_title_feature)
        document_title_feature = document_title_feature[:, 0, :]

        passage_embedding = self.pretrain_model(input_ids=passages_input_ids).last_hidden_state
        passage_embedding = self.dropout(passage_embedding)
        passage_feature = passage_embedding[:, 0, :]

        span_indices = torch.where(passages_input_ids == 30522)
        span_feature = torch.index_select(passage_embedding, 0, span_indices[0])
        span_feature = span_feature.gather(1, span_indices[1].unsqueeze(-1))
        span_feature = span_feature.view(-1, 768)

        x = torch.concat((document_title_feature, passage_feature, span_feature), dim=0)
        x = self.conv1(x, edge_index, edge_type)
        x = self.conv2(x, edge_index, edge_type)
        x = self.classifier(x).view(-1)

        # document_score = x[0]
        # title_score = x[1:len(document_title_feature)]
        # passage_score = x[len(document_title_feature):len(document_title_feature) + len(passage_feature)]
        span_score = torch.exp(x[-len(span_feature):] / self.tau)

        if labels:
            # calculate loss
            loss = -torch.log(torch.sum(span_score[labels]) / torch.sum(span_score))
            return loss, span_score
        return span_score


class StructureRerankDataset(Dataset):
    def __init__(self, ids_list, contexts_list, labels_list):
        self.ids_list = ids_list
        self.contexts_list = contexts_list
        self.labels_list = labels_list

    def __len__(self):
        return len(self.contexts_list)

    def __getitem__(self, index):
        ids = self.ids_list[index]
        contexts = self.contexts_list[index]
        labels = self.labels_list[index]
        return ids, contexts, labels


class StructureRerankRunner(BaseRunner):
    def __init__(self, args):
        super(StructureRerankRunner, self).__init__()
        pretrained_path = f'{args.home}/{args.code}/pretrained/{args.model}/'
        if not os.path.exists(pretrained_path):
            utils.mk_dir(pretrained_path)
            tokenizer = AutoTokenizer.from_pretrained(args.model)
            config = AutoConfig.from_pretrained(args.model)
            config.dropout_rate = args.dropout
            pretrain_model = AutoModel.from_pretrained(args.model, config=config)
            special_tokens = ['[SPN]']
            tokenizer.add_tokens(special_tokens)
            pretrain_model.resize_token_embeddings(len(tokenizer))
            tokenizer.save_pretrained(pretrained_path)
            config.save_pretrained(pretrained_path)
            pretrain_model.save_pretrained(pretrained_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
            config = AutoConfig.from_pretrained(pretrained_path)
            config.dropout_rate = args.dropout
            pretrain_model = AutoModel.from_pretrained(pretrained_path, config=config)
        model = StructureRerankModel(pretrain_model, args.dropout, args.tau)
        self.args = args
        self.tokenizer = tokenizer
        self.model = model

    def prepare_dataset(self, args, split='train'):
        ids_list, contexts_list, labels_list = [], [], []

        # TODO

        # create dataset
        dataset = StructureRerankDataset(ids_list, contexts_list, labels_list)
        return dataset

    def forward(self, model, payload, tokenizer, device):
        ids, document_titles, passages, edge_index, edge_type, labels = payload
        document_title_input_ids = tokenizer.batch_encode_plus(
            document_titles, padding=True, truncation=True, return_tensors='pt',
            max_length=self.args.source_sequence_size
        ).input_ids.to(device)
        passages_input_ids = tokenizer.batch_encode_plus(
            passages, padding=True, truncation=True, return_tensors='pt',
            max_length=self.args.source_sequence_size
        ).input_ids.to(device)

        outputs = model(
            document_title_input_ids, passages_input_ids, edge_index.to(device), edge_type.to(device), labels.to(device)
        )
        return outputs

    def inference(self, model, payload, tokenizer, device, result_dict):
        ids, (document_titles, passages, edge_index, edge_type), labels = payload
        document_title_input_ids = tokenizer.batch_encode_plus(
            document_titles, padding=True, truncation=True, return_tensors='pt',
            max_length=self.args.source_sequence_size
        ).input_ids.to(device)
        passages_input_ids = tokenizer.batch_encode_plus(
            passages, padding=True, truncation=True, return_tensors='pt',
            max_length=self.args.source_sequence_size
        ).input_ids.to(device)

        outputs = model(
            document_title_input_ids, passages_input_ids, edge_index.to(device), edge_type.to(device)
        )

        result_dict['id_list'].append(ids)
        index_score_pair = [(index, score) for index, score in zip(range(len(outputs)), outputs.item())]
        index_score_pair.sort(key=lambda x: x[1], reverse=True)
        result_dict['outputs'].append([x[0] for x in index_score_pair])
        result_dict['targets'].append(labels.item())

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
            'r@1': AverageMeter('R@1', ':6.4f'),
            'r@5': AverageMeter('R@5', ':6.4f'),
            'r@10': AverageMeter('R@10', ':6.4f'),
            'r@50': AverageMeter('R@10', ':6.4f'),
        }
        return copy.deepcopy(meters)

    def measure_result(self, result_dict, meters):
        count_dict_template = {f'{k}': 0 for k in [1, 5, 10, 50]}
        for output, target in zip(result_dict['outputs'], result_dict['targets']):
            count_dict = copy.deepcopy(count_dict_template)
            for t in target:
                for k in count_dict:
                    if t in output[:int(k)]:
                        count_dict[k] += 1

            for k, v in count_dict:
                meters[f'r@{k}'].update(count_dict[k] / len(target))
