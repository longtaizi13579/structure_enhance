import json
import copy
import random
from sklearn.metrics import f1_score, accuracy_score

import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import BertModel, BertTokenizer

from abstracts import BaseRunner
from utils import AverageMeter


class TransformerSpanPredictionModel(nn.Module):
    def __init__(self, bert, dropout=0.1, alpha=0.5):
        super(TransformerSpanPredictionModel, self).__init__()
        self.bert = bert
        self.loss_func = nn.BCELoss()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(768, 1),
            nn.Sigmoid()
        )
        self.alpha = alpha

    def forward(self, input_ids, labels=None, device='cpu'):
        batch_size = len(input_ids)
        u = self.bert(input_ids=input_ids).last_hidden_state
        u = self.dropout(u)

        output = self.classifier(u)
        output = output.view(batch_size, -1)
        cls_output = output[:, 0]

        spn_indices = torch.where(input_ids == 30522)
        spn_output = torch.index_select(output, 0, spn_indices[0])
        spn_output = spn_output.gather(1, spn_indices[1].unsqueeze(-1))
        spn_output = spn_output.view(-1)

        if labels:
            # section loss
            cls_target = torch.Tensor([0 if sum(x) == 0 else 1 for x in labels]).to(device)
            loss_section = self.loss_func(cls_output, cls_target) * self.alpha

            # span loss
            labels = torch.Tensor([x for lst in labels for x in lst]).to(device)
            spn_target = labels.view(-1)
            loss_span = self.loss_func(spn_output, spn_target)
            loss = loss_span + loss_section
            return loss, loss_section, loss_span, cls_output, spn_output

        return cls_output, spn_output


class TransformerSpanPredictionDataset(Dataset):
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


class TransformerRunner(BaseRunner):
    def __init__(self, args):
        super(TransformerRunner, self).__init__()
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        special_token_dict = {'additional_special_tokens': ['[SPN]']}
        tokenizer.add_special_tokens(special_token_dict)
        bert = BertModel.from_pretrained(f"bert-base-uncased")
        bert.resize_token_embeddings(len(tokenizer))
        self.model = TransformerSpanPredictionModel(bert, args.dropout)
        self.tokenizer = tokenizer

    @staticmethod
    def collate(batch):
        ids = [item[0] for item in batch]
        context = [item[1] for item in batch]
        label = [item[2] for item in batch]
        return [ids, context, label]

    def prepare_dataset(self, args, split='train'):
        id_list = []
        context_list = []
        label_list = []
        # load section-span map
        section_map = {}
        with open(f'{args.home}/multidoc2dial/multidoc2dial_doc.json', 'r') as f:
            doc_data = json.load(f)['doc_data']
            for domain, docs in doc_data.items():
                for doc_id, content in docs.items():
                    section_map[doc_id] = {}
                    for span_id, span in content['spans'].items():
                        span_list = section_map[doc_id].get(span['id_sec'], [])
                        span_list.append((span_id, span['text_sp']))
                        section_map[doc_id][span['id_sec']] = span_list

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
                        all_text = [x[1] for x in section_map[doc_id][sec_id]]
                        all_id = [x[0] for x in section_map[doc_id][sec_id]]
                        if len(all_text) > 10:
                            continue
                        section = ' [SPN] '.join(all_text)
                        label = [1 if x in span_list else 0 for x in all_id]
                        context = ' [SPN] ' + section + ' [SEP] ' + question_str
                        id_list.append(dialog_id + '_' + str(next_turn['turn_id']))
                        context_list.append(context)
                        label_list.append(label)

                    # negative
                    sec_id = random.choice(list(section_map[doc_id].keys()))
                    if sec_id not in groundings:
                        all_text = [x[1] for x in section_map[doc_id][sec_id]]
                        if len(all_text) > 10:
                            continue
                        section = ' [SPN] '.join(all_text)
                        context = ' [SPN] ' + section + ' [SEP] ' + question_str
                        id_list.append(dialog_id + '_' + str(next_turn['turn_id']))
                        context_list.append(context)
                        label_list.append([0 for x in all_text])

        # create dataset
        dataset = TransformerSpanPredictionDataset(id_list, context_list, label_list)
        return dataset

    def forward(self, model, payload, tokenizer, device):
        ids, context, label = payload
        input_ids = tokenizer.batch_encode_plus(
            list(context), max_length=512, truncation=True, padding=True, return_tensors='pt').input_ids.to(device)
        # calculate loss
        outputs = model(input_ids=input_ids, labels=label, device=device)
        return outputs

    def inference(self, model, payload, tokenizer, device, result_dict):
        ids, context, labels = payload
        input_ids = tokenizer.batch_encode_plus(
            list(context), max_length=512, truncation=True, padding=True, return_tensors='pt').input_ids.to(device)
        cls_output, spn_output = model(input_ids, device=device)

        result_dict['id_list'] += ids
        result_dict['outputs']['section'] += [1 if x > 0.5 else 0 for x in cls_output.tolist()]
        result_dict['targets']['section'] += [0 if sum(x) == 0 else 1 for x in labels]

        spn_output = spn_output.tolist()
        i = 0
        for lab in labels:
            result_dict['outputs']['span'].append([1 if x > 0.5 else 0 for x in spn_output[i:i + len(lab)]])
            result_dict['targets']['span'].append(lab)
            i += len(lab)

    def init_losses_dict(self):
        losses = {
            'loss': AverageMeter('Loss', ':.4e'),
            'loss_sec': AverageMeter('Section Loss', ':.4e'),
            'loss_spn': AverageMeter('Span Loss', ':.4e'),
        }
        return copy.deepcopy(losses)

    def measure_loss(self, payload, model_output, losses):
        loss, loss_section, loss_span, cls_output, spn_output = model_output
        losses['loss'].update(loss.item(), len(cls_output))
        losses['loss_sec'].update(loss_section.item(), len(cls_output))
        losses['loss_spn'].update(loss_span.item(), len(cls_output))

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
