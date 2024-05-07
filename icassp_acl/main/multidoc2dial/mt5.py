import json
import copy
import sacrebleu
from rouge import Rouge
from torch.utils.data import Dataset
from transformers import MT5Tokenizer, MT5ForConditionalGeneration

import utils
from utils import AverageMeter
from abstracts import BaseRunner


class ResponseGenerationDataset(Dataset):
    def __init__(self, query_list, context_list, label_list):
        self.query_list = query_list
        self.context_list = context_list
        self.label_list = label_list

    def __len__(self):
        return len(self.context_list)

    def __getitem__(self, index):
        query = self.query_list[index]
        context = self.context_list[index]
        label = self.label_list[index]
        return query, context, label


class MT5Runner(BaseRunner):
    def __init__(self, args):
        super(MT5Runner, self).__init__()
        self.args = args
        self.tokenizer = MT5Tokenizer.from_pretrained("google/mt5-base")
        self.model = MT5ForConditionalGeneration.from_pretrained(f"google/mt5-base")
        special_tokens = ["<last_turn>", "<user>", "<agent>", "<response>", "<passage>"]
        self.tokenizer.add_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))

    def prepare_dataset(self, args, split='train'):
        query_list = []
        context_list = []
        label_list = []
        with open(f'{args.home}/g4/workshop-{split}.jsonl', 'r') as f:
            for line in f.readlines():
                sample = json.loads(line)
                query_list.append(sample['query'])
                context_list.append(json.loads(sample['rerank'])[0])
                label_list.append(sample['response'])

        # create dataset
        dataset = ResponseGenerationDataset(query_list, context_list, label_list)
        return dataset

    def forward(self, model, payload, tokenizer, device):
        query, context, label = payload
        input_ids = tokenizer.batch_encode_plus(
            list(context), padding=True, return_tensors='pt').input_ids.to(device)
        label_ids = tokenizer.batch_encode_plus(
            list(label), padding=True, return_tensors='pt').input_ids.to(device)
        outputs = model(input_ids=input_ids, labels=label_ids)
        return outputs

    def inference(self, model, payload, tokenizer, device, result_dict):
        query, context, label = payload
        input_ids = tokenizer.batch_encode_plus(list(context), padding=True, return_tensors='pt').input_ids.to(device)
        outputs = model.generate(input_ids, num_beams=3, max_length=128, early_stopping=True, no_repeat_ngram_size=3)
        hypothesis = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        label = tokenizer.batch_decode(tokenizer.batch_encode_plus(
            label, add_special_tokens=False).input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
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
        hypothesis_list = result_dict['outputs']
        hypothesis_list = [x if len(x) > 10 else 'placeholder' for x in hypothesis_list]
        reference_list = result_dict['targets']
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
