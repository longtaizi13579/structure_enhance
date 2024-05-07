import json
import copy
import sacrebleu
from rouge import Rouge
from torch.utils.data import Dataset
from transformers import BartTokenizer, BartForConditionalGeneration

import utils
from utils import AverageMeter
from abstracts import BaseRunner


class ResponseGenerationDataset(Dataset):
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


class BartRunner(BaseRunner):
    def __init__(self):
        super(BartRunner, self).__init__()
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        self.model = BartForConditionalGeneration.from_pretrained(f"facebook/bart-base")

    def prepare_dataset(self, args, split='train'):
        id_list = []
        context_list = []
        label_list = []
        # load passage map
        id_psg_map = {}
        with open(f'{args.home}/g4/mdd_passages.jsonl', 'r') as f:
            for line in f.readlines():
                sample = json.loads(line)
                id_psg_map[sample['pid']] = sample['text']

        if split != 'test':
            # process dialog data
            with open(f'{args.home}/g4/mdd-{split}-kilt.jsonl', 'r') as f:
                for line in f.readlines():
                    # load sample
                    sample = json.loads(line)
                    grounding = id_psg_map[sample['output'][0]['provenance'][0]['wikipedia_id']]
                    history = sample['input']
                    label = sample['output'][0]['answer']

                    # package data
                    id_list.append(sample['id'])
                    context = ' '.join(grounding.split(' ')[:400]) + '[SEP]' + ' '.join(history.split(' ')[:100])
                    context_list.append(context)
                    label_list.append(label)
        else:
            # process dialog data
            with open(f'{args.home}/g4/dev-test.jsonl', 'r') as f:
                for line in f.readlines():
                    # load sample
                    sample = json.loads(line)
                    grounding = id_psg_map[sample['scored_pids'][0][0]]

                    # package data
                    id_list.append(sample['id'])
                    context = ' '.join(grounding.split(' ')[:400]) + '[SEP]' + ' '.join(
                        sample['input'].split(' ')[:100])
                    context_list.append(context)
                    label_list.append('')

        # create dataset
        dataset = ResponseGenerationDataset(id_list, context_list, label_list)
        return dataset

    def forward(self, model, payload, tokenizer, device):
        ids, context, label = payload
        input_ids = tokenizer.batch_encode_plus(
            list(context), padding=True, return_tensors='pt').input_ids.to(device)
        label_ids = tokenizer.batch_encode_plus(
            list(label), padding=True, return_tensors='pt').input_ids.to(device)
        outputs = model(input_ids=input_ids, labels=label_ids)
        return outputs

    def inference(self, model, payload, tokenizer, device, result_dict):
        ids, context, label = payload
        input_ids = tokenizer.batch_encode_plus(list(context), padding=True, return_tensors='pt').input_ids.to(device)
        outputs = model.generate(input_ids, num_beams=4, max_length=128, early_stopping=True, no_repeat_ngram_size=3)
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
        hypothesis_list = result_dict['outputs']
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
