import os
import re
import string
import shutil
import torch
from datetime import datetime
from collections import Counter
from contextlib import contextmanager
from transformers import AdamW, get_scheduler
from torch.distributed import barrier


class Logger:
    def __init__(self, file_logger, outputs='file'):
        self.file_logger = file_logger
        self.outputs = outputs

    def info(self, x):
        if self.outputs in ['both', 'terminal']:
            print(x)
        if self.outputs in ['both', 'file']:
            self.file_logger.info(x)


def mk_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def parameter_number(model):
    total = sum([param.nelement() for param in model.parameters()])
    print(f'Number of parameter:{round(total / 1e6)}M')


# Dataset
def load_dataset(runner, dataset_path, split, args):
    with torch_distributed_main_first(args.local_rank):
        data_file_path = os.path.join(dataset_path, split)
        if not os.path.exists(data_file_path) or args.reload:
            dataset = runner.prepare_dataset(args, split)
            args.logger.info(f"{split.capitalize()} Dataset Created: {datetime.now()} \t Length: {len(dataset)}")
            torch.save(dataset, data_file_path)
        else:
            dataset = torch.load(data_file_path)

    if args.debug:
        args.logger.info(f'{split.capitalize()} Dataset: \n {dataset[0]}')
    return dataset


# Pretrain
def load_pretrain(args, config_cla, tokenizer_cla, model_cla, model_name=None):
    if model_name is None:
        model_name = args.model

    pretrained_path = f'{args.pretrained_path}/{model_name}/'
    if os.path.exists(pretrained_path):
        if config_cla:
            config = config_cla.from_pretrained(pretrained_path)
            if args.dropout:
                config.dropout_rate = args.dropout
        else:
            config = None
        model = model_cla.from_pretrained(pretrained_path, config=config)
        tokenizer = tokenizer_cla.from_pretrained(pretrained_path)
    else:
        if config_cla:
            config = config_cla.from_pretrained(model_name)
            if args.dropout:
                config.dropout_rate = args.dropout
            if args.num_labels:
                config.num_labels = args.num_labels
            config.save_pretrained(pretrained_path)
        else:
            config = None
        model = model_cla.from_pretrained(model_name, config=config)
        tokenizer = tokenizer_cla.from_pretrained(model_name)
        tokenizer.save_pretrained(pretrained_path)
        model.save_pretrained(pretrained_path)

    return config, tokenizer, model


# Checkpoint
def save_checkpoint(state, is_best, dir):
    torch.save(state, dir + '/checkpoint.pth.tar')
    if is_best:
        shutil.copyfile(dir + '/checkpoint.pth.tar', dir + '/model_best.pth.tar')


def display_checkpoint_info(checkpoint, path, args):
    fmt = 'Checkpoint Path: {} \t Epoch: {:d} \t Best Score: {:2f} \t Time: {}'
    args.logger.info(fmt.format(path,
                                checkpoint['epoch'],
                                checkpoint['best_score'],
                                datetime.fromtimestamp(checkpoint['time']).strftime('%Y-%m-%d %H:%M:%S')))


# Meter
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.count = 0
        self.sum = 0
        self.avg = 0
        self.val = 0
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, logger, num_batches, meters, prefix=''):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.logger = logger

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        self.logger.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def matching_evaluate(references, predictions):
    f1 = em = total = 0
    for ref_text, prediction in zip(references, predictions):
        total += 1
        ground_truths = [ref_text]
        f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)
        em += metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
    f1 = 100.0 * f1 / total
    em = 100.0 * em / total

    return f1, em


# Context
@contextmanager
def torch_distributed_main_first(local_rank: int):
    if local_rank not in [-1, 0]:
        barrier()

    yield

    if local_rank == 0:
        barrier()


# Optimizer
def prepare_optimizer(model, args):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        }
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.lr, eps=args.eps
    )
    return optimizer


def prepare_scheduler(optimizer, args, steps_per_epoch):
    total_steps = args.epochs * steps_per_epoch
    if args.warmup_rate:
        warmup_steps = int(total_steps * args.warmup_rate)
    else:
        warmup_steps = args.warmup_steps
    args.logger.info(
        f"Prepare Scheduler: {args.scheduler} \t warm_up_steps: {warmup_steps} \t total: {total_steps}")
    scheduler = get_scheduler(name=args.scheduler, optimizer=optimizer, num_warmup_steps=warmup_steps,
                              num_training_steps=total_steps)
    return scheduler
