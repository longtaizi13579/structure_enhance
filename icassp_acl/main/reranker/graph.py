import json
import copy
import random
from sklearn.metrics import f1_score, accuracy_score

import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import BertModel, BertTokenizer

from torch_geometric.nn import RGCNConv
from abstracts import BaseRunner
from utils import AverageMeter


class InputSample:

    def __init__(self):
        self.edge_index = []
        self.edge_type = []
        # document 节点默认为第一个
        self.node_tet = []
        self.parent_child_map = {
            '1': []
        }
        self.label = []


class GraphSpanPredictionDataset(Dataset):
    def __init__(self, id_list, context_list, edge_index_list, edge_type_list, label_list):
        self.id_list = id_list
        self.context_list = context_list
        self.edge_index_list = edge_index_list
        self.edge_type_list = edge_type_list
        self.label_list = label_list

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, index):
        id_ = self.id_list[index]
        context = self.context_list[index]
        edge_index = self.edge_index_list[index]
        edge_type = self.edge_type_list[index]
        label = self.label_list[index]
        return id_, context, edge_index, edge_type, label


class GraphSpanPredictionModel(nn.Module):
    def __init__(self, bert, dropout, alpha=0.5):
        super(GraphSpanPredictionModel, self).__init__()
        self.bert = bert
        self.conv1 = RGCNConv(768, 768, 2)
        self.conv2 = RGCNConv(768, 768, 2)
        self.classifier = nn.Sequential(
            nn.Linear(768, 1),
            nn.Sigmoid()
        )
        self.loss_func = nn.BCELoss()
        self.dropout = nn.Dropout(dropout)
        self.alpha = alpha

    def forward(self, input_ids, edge_index, edge_type, labels=None, device='cpu'):
        x = self.bert(input_ids=input_ids).pooler_output
        x = self.dropout(x)
        # x = self.conv1(x, edge_index, edge_type)
        # x = self.conv2(x, edge_index, edge_type)
        x = self.classifier(x).view(-1)
        sec_output = x[1].unsqueeze(-1)
        spn_output = x[2:]

        if labels:
            sec_target = torch.tensor([labels[0]], dtype=torch.float).to(device)
            loss_section = self.loss_func(sec_output, sec_target) * self.alpha

            spn_target = torch.tensor(labels[1:], dtype=torch.float).to(device)
            loss_span = self.loss_func(spn_output, spn_target)
            loss = loss_span + loss_section

            return loss, loss_section, loss_span

        return sec_output, spn_output


class GraphRunner(BaseRunner):

    def __init__(self, args):
        super(GraphRunner, self).__init__()
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        bert = BertModel.from_pretrained(f"bert-base-uncased")
        self.model = GraphSpanPredictionModel(bert, args.dropout)
        self.tokenizer = tokenizer
        self.graph_dict = {}

    def prepare_dataset(self, args, split='train'):
        id_list = []
        context_list = []
        edge_index_list = []
        edge_type_list = []
        label_list = []
        # construct section map
        section_map = {}
        with open(f'{args.home}/multidoc2dial/multidoc2dial_doc.json', 'r') as f:
            doc_data = json.load(f)['doc_data']
            for domain, docs in doc_data.items():
                for doc_id, content in docs.items():
                    section_map[doc_id] = {}
                    for span_id, span in content['spans'].items():
                        span_dict = section_map[doc_id].get(span['id_sec'], {'title': span['title'], 'spans': []})
                        span_dict['spans'].append((span_id, span['text_sp']))
                        section_map[doc_id][span['id_sec']] = span_dict
        # load data
        if split == 'train':
            with open(f'{args.home}/multidoc2dial/multidoc2dial_dial_train.json', 'r') as f:
                dial_data = json.load(f)['dial_data']
        elif split == 'valid':
            with open(f'{args.home}/multidoc2dial/multidoc2dial_dial_validation.json', 'r') as f:
                dial_data = json.load(f)['dial_data']
        elif split == 'test':
            with open(f'{args.home}/multidoc2dial/multidoc2dial_dial_test.json', 'r') as f:
                dial_data = json.load(f)['dial_data']
        else:
            raise Exception(f'Undefined split type: {split}')

        # process dialog
        for domain, dialogs in dial_data.items():
            for dialog in dialogs:
                dialog_id = dialog['dial_id']
                turns = dialog['turns']
                all_prev_utterances = []
                for turn in turns:
                    utterance_line = turn["utterance"].replace("\n", " ").replace("\t", " ")
                    all_prev_utterances.append("{}: {}".format(turn["role"], utterance_line))
                    question_str = utterance_line + "[SEP]" + "||".join(reversed(all_prev_utterances[:-1]))

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

                    groundings = {}
                    for x in next_turn['references']:
                        doc_id = x['doc_id']
                        span_id = x['id_sp']
                        span = doc_data[domain][doc_id]['spans'][span_id]
                        span_list = groundings.get(span['id_sec'], [])
                        span_list.append(span_id)
                        groundings[span['id_sec']] = span_list

                    # positive
                    for sec_id, span_list in groundings.items():
                        # context
                        section = section_map[doc_id][sec_id]
                        title = section['title']
                        span_text = [x[1] for x in section['spans']]
                        if len(span_text) > 10:
                            continue
                        section_text = ' '.join(span_text)
                        context = [title, section_text] + [x + '[SEP]' + section_text for x in span_text]
                        context = [x + '[SEP]' + question_str for x in context]
                        context_list.append(context)

                        # id
                        id_list.append(dialog_id + '_' + str(next_turn['turn_id']))

                        # edge index list
                        edge_index = [
                            [0, 1] + [1] * len(span_text) + [i + 2 for i in range(len(span_text))],
                            [1, 0] + [i + 2 for i in range(len(span_text))] + [1] * len(span_text)
                        ]
                        edge_index_list.append(edge_index)

                        # edge type list
                        edge_type = [0, 1] + [0] * len(span_text) + [1] * len(span_text)
                        edge_type_list.append(edge_type)

                        # label list
                        span_id = [x[0] for x in section['spans']]
                        label = [1] + [1 if x in span_list else 0 for x in span_id]
                        label_list.append(label)

                    # negative
                    sec_id = random.choice(list(section_map[doc_id].keys()))
                    if sec_id not in groundings:
                        # context
                        section = section_map[doc_id][sec_id]
                        title = section['title']
                        span_text = [x[1] for x in section['spans']]
                        if len(span_text) > 10:
                            continue
                        section_text = ' '.join(span_text)
                        context = [title, section_text] + [x + '[SEP]' + section_text for x in span_text]
                        context = [x + '[SEP]' + question_str for x in context]
                        context_list.append(context)

                        # id
                        id_list.append(dialog_id + '_' + str(next_turn['turn_id']))

                        # edge index list
                        edge_index = [
                            [0, 1] + [1] * len(span_text) + [i + 2 for i in range(len(span_text))],
                            [1, 0] + [i + 2 for i in range(len(span_text))] + [1] * len(span_text)
                        ]
                        edge_index_list.append(edge_index)

                        # edge type list
                        edge_type = [0, 1] + [0] * len(span_text) + [1] * len(span_text)
                        edge_type_list.append(edge_type)

                        # label list
                        label = [0] * (len(span_text) + 1)
                        label_list.append(label)

        # create dataset
        dataset = GraphSpanPredictionDataset(id_list, context_list, edge_index_list, edge_type_list, label_list)
        return dataset

    def collate(self, batch):
        ids = [item[0] for item in batch]
        context = [item[1] for item in batch]
        edge_index = [item[2] for item in batch]
        edge_type = [item[3] for item in batch]
        label = [item[4] for item in batch]
        return [ids, context, edge_index, edge_type, label]

    def forward(self, model, payload, tokenizer, device):
        batch_size = len(payload[0])
        total_loss = torch.tensor(0, dtype=torch.float).to(device)
        total_loss_section = torch.tensor(0, dtype=torch.float).to(device)
        total_loss_span = torch.tensor(0, dtype=torch.float).to(device)
        for id, context, edge_index, edge_type, label in \
                zip(payload[0], payload[1], payload[2], payload[3], payload[4]):
            edge_index = torch.tensor(edge_index, dtype=torch.long).to(device)
            edge_type = torch.tensor(edge_type, dtype=torch.long).to(device)
            input_ids = tokenizer.batch_encode_plus(
                list(context), max_length=512, truncation=True, padding=True, return_tensors='pt').input_ids.to(device)
            loss, loss_section, loss_span = model(input_ids, edge_index, edge_type, label, device)
            total_loss += loss
            total_loss_section += loss_section
            total_loss_span += loss_span
        total_loss /= batch_size
        total_loss_section /= batch_size
        total_loss_span /= batch_size
        return total_loss, total_loss_section, total_loss_span

    def inference(self, model, payload, tokenizer, device, result_dict):
        for id, context, edge_index, edge_type, label in \
                zip(payload[0], payload[1], payload[2], payload[3], payload[4]):
            edge_index = torch.tensor(edge_index, dtype=torch.long).to(device)
            edge_type = torch.tensor(edge_type, dtype=torch.long).to(device)
            input_ids = tokenizer.batch_encode_plus(
                list(context), max_length=512, truncation=True, padding=True, return_tensors='pt').input_ids.to(device)
            sec_output, spn_output = model(input_ids, edge_index, edge_type, None, device)
            result_dict['id_list'].append(id)
            result_dict['outputs']['section'].append(1 if sec_output.tolist()[0] > 0.5 else 0)
            result_dict['outputs']['span'].append([1 if x > 0.5 else 0 for x in spn_output.tolist()])
            result_dict['targets']['section'].append(label[0])
            result_dict['targets']['span'].append(label[1:])

    def init_losses_dict(self):
        losses = {
            'loss': AverageMeter('Loss', ':.4e'),
            'loss_sec': AverageMeter('Section Loss', ':.4e'),
            'loss_spn': AverageMeter('Span Loss', ':.4e'),
        }
        return copy.deepcopy(losses)

    def measure_loss(self, payload, model_output, losses):
        id, context, edge_index, edge_type, label = payload
        loss, loss_section, loss_span = model_output
        losses['loss'].update(loss.item(), len(id))
        losses['loss_sec'].update(loss_section.item(), len(id))
        losses['loss_spn'].update(loss_span.item(), len(id))

    def init_results_dict(self):
        results = {
            'id_list': [],
            'outputs': {
                'section': [],
                'span': []
            },
            'targets': {
                'section': [],
                'span': []
            }
        }
        return copy.deepcopy(results)

    def init_meters_dict(self):
        meters = {
            'f1_sec': AverageMeter('Section F1', ':6.4f'),
            'acc_spn': AverageMeter('Span Acc', ':6.4f'),
        }
        return copy.deepcopy(meters)

    def measure_result(self, result_dict, meters):
        instance_num = len(result_dict['id_list'])
        targets = result_dict['targets']
        outputs = result_dict['outputs']

        meters['f1_sec'].update(f1_score(targets['section'], outputs['section']), instance_num)
        meters['acc_spn'].update(accuracy_score(
            [str(x) for x in targets['span']], [str(x) for x in outputs['span']]
        ), instance_num)
