import os
import json
import time
import logging
import datetime
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel

import utils
from options import Options
from utils import Logger, AverageMeter, ProgressMeter, torch_distributed_main_first
from reranker.transformer import TransformerRunner
# from reranker.graph import GraphRunner
from multidoc2dial.bart import BartRunner
from multidoc2dial.unigdd import UniGddRunner
from universe.unigdd import UniverseUniGddRunner
from multidoc2dial.fid import FiDRunner
from reranker.longformer import LongFormerRunner
from reranker.treejc import TreeJCRunner
from retriever.dpr import DPRRunner
from generator.rag import RAGRunner
from generator.struc_fid import StrucFiDRunner
from multidoc2dial.mt5 import MT5Runner
from doc2bot import PolicyRunner, ActRunner, SFRe2GRunner


def main_work(args, runner):
    tokenizer = runner.tokenizer
    model = runner.model

    args.checkpoint_path = os.path.join(args.home, args.code, args.checkpoint_dir, args.name)
    logger.info(json.dumps(vars(args), indent=2))
    args.logger = logger

    writer = SummaryWriter(os.path.join(args.checkpoint_path, 'tensorboard'))

    # set up train device
    if args.n_gpu > 1:
        device = torch.device("cuda", args.local_rank)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() and args.n_gpu > 0 else 'cpu')
    model.to(device)

    dataset_path = os.path.join(args.home, args.code, args.dataset_prefix + args.dataset)
    utils.mk_dir(dataset_path)
    collate = runner.collate if hasattr(runner, 'collate') else None

    # train
    if args.mode == 'train':
        best_score = 0
        # load dataset
        train_dataset = utils.load_dataset(runner, dataset_path, 'train', args)
        valid_dataset = utils.load_dataset(runner, dataset_path, 'valid', args)

        # prepare dataloader
        sampler = DistributedSampler(train_dataset) if args.n_gpu > 1 else None
        shuffle = True if sampler is None else False
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.per_gpu_batch_size, shuffle=shuffle,
                                  collate_fn=collate, sampler=sampler)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.eval_batch_size, collate_fn=collate)

        # prepare criterion & optimizer & scheduler
        optimizer = utils.prepare_optimizer(model, args)
        steps_per_epoch = len(train_loader) // args.accumulation_steps
        scheduler = utils.prepare_scheduler(optimizer, args, steps_per_epoch)

        # continue with checkpoint
        if os.path.exists(args.checkpoint_path + '/checkpoint.pth.tar'):
            checkpoint = torch.load(args.checkpoint_path + '/checkpoint.pth.tar', map_location='cpu')
            utils.display_checkpoint_info(checkpoint, args.checkpoint_path + '/checkpoint.pth.tar', args)
            args.start_epoch = checkpoint['epoch']
            tokenizer = checkpoint['tokenizer']
            model.load_state_dict(checkpoint['state_dict'])
            best_score = checkpoint['best_score']
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
        elif args.init_state_dict_from:
            init_checkpoint_path = os.path.join(args.home, args.code, args.checkpoint_dir, args.init_state_dict_from)
            checkpoint = torch.load(init_checkpoint_path + '/model_best.pth.tar', map_location='cpu')
            logger.info('Initialize Model State Dict From:')
            utils.display_checkpoint_info(checkpoint, init_checkpoint_path + '/model_best.pth.tar', args)
            model.load_state_dict(checkpoint['state_dict'])

        model.to(device)
        if args.n_gpu > 1:
            model = DistributedDataParallel(
                model,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=args.no_find_unused_parameters
            )

        # train
        for epoch in range(args.start_epoch, args.epochs):
            # prepare dataloader
            if args.n_gpu > 1:
                assert args.reload_train_per_epoch is False
                sampler.set_epoch(epoch)
            train(train_loader, tokenizer, model, optimizer, scheduler, epoch, device, args, writer)
            if args.local_rank in [-1, 0]:
                if isinstance(model, DistributedDataParallel):
                    eval_model = model.module
                else:
                    eval_model = model

                if args.skip_eval_dur_train or (epoch < args.start_eval_epoch):
                    score = 0
                    is_best = True
                else:
                    meters = validate(valid_loader, tokenizer, eval_model, epoch, device, args, writer)
                    # check best
                    score = runner.score_func(meters)
                    is_best = score > best_score
                    best_score = max(best_score, score)
                if not args.debug:
                    # save checkpoint
                    writer.add_scalars('score', {'score': score}, epoch)
                    utils.save_checkpoint({
                        'epoch': epoch + 1,
                        'best_score': score,
                        'time': time.time(),
                        'args': args,
                        'tokenizer': tokenizer,
                        'state_dict': eval_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict()
                    }, is_best, args.checkpoint_path)

            if args.n_gpu > 1:
                torch.distributed.barrier()

            if args.reload_train_per_epoch:
                train_dataset = runner.reload_train_dataset_per_epoch(args, epoch + 1)
                sampler = DistributedSampler(train_dataset) if args.n_gpu > 1 else None
                shuffle = True if sampler is None else False
                train_loader = DataLoader(dataset=train_dataset, batch_size=args.per_gpu_batch_size, shuffle=shuffle,
                                          collate_fn=collate, sampler=sampler)
            if args.debug:
                break
    elif args.mode == 'valid':
        # load valid dataset
        valid_dataset = utils.load_dataset(runner, dataset_path, 'valid', args)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.eval_batch_size, collate_fn=collate)

        # load model
        checkpoint_path = args.checkpoint_path + '/model_best.pth.tar'
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            utils.display_checkpoint_info(checkpoint, checkpoint_path, args)
            tokenizer = checkpoint['tokenizer']
            model.load_state_dict(checkpoint['state_dict'])
            model.to(device)

        # validate
        validate(valid_loader, tokenizer, model, None, device, args, writer)
    elif args.mode == 'test':
        # load test dataset
        test_dataset = utils.load_dataset(runner, dataset_path, 'test', args)
        test_loader = DataLoader(dataset=test_dataset, batch_size=args.eval_batch_size, collate_fn=collate)

        # load model
        checkpoint_path = args.checkpoint_path + '/model_best.pth.tar'
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        utils.display_checkpoint_info(checkpoint, checkpoint_path, args)
        tokenizer = checkpoint['tokenizer']
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        # test
        test(test_loader, tokenizer, model, device, args)
    else:
        raise Exception(f'Undefined run mode: {args.mode}')
    writer.close()

    if args.n_gpu > 1:
        destroy_process_group()


def train(train_loader, tokenizer, model, optimizer, scheduler, epoch, device, args, writer):
    batch_time = AverageMeter('Time', ':6.3f')
    learning_rate = AverageMeter('Lr', ':.4e')
    losses = runner.init_losses_dict()
    progress = ProgressMeter(
        logger,
        len(train_loader),
        [batch_time, learning_rate] + list(losses.values()),
        prefix="Train Epoch: [{}]".format(epoch)
    )
    model.train()

    start_time = time.time()

    for index, payload in enumerate(tqdm(train_loader)):
        # compute output
        outputs = runner.forward(model, payload, tokenizer, device)

        # update losses
        runner.measure_loss(payload, outputs, losses)

        # compute gradient
        loss = outputs[0]
        if args.n_gpu > 1:
            loss = loss.mean()

        if args.accumulation_steps > 1:
            loss = loss / args.accumulation_steps

        loss.backward()

        if (index + 1) % args.accumulation_steps == 0:
            if args.max_grad_norm is not None and args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # measure batch time
        batch_time.update(time.time() - start_time)
        start_time = time.time()
        if (index + 1) % args.print_freq == 0:
            # measure learning rate
            learning_rate.update(optimizer.state_dict()['param_groups'][0]['lr'])
            progress.display(index + 1)

        if args.debug:
            break

    if not args.debug:
        writer.add_scalars('loss train', {field_name: loss.avg for field_name, loss in losses.items()}, epoch)


def validate(valid_loader, tokenizer, model, epoch, device, args, writer):
    batch_time = AverageMeter('Time', ':6.3f')
    meters = runner.init_meters_dict()
    progress = ProgressMeter(
        logger,
        len(valid_loader),
        [batch_time] + list(meters.values()),
        prefix='Valid Epoch: [{}]'.format(epoch)
    )

    result_dict = runner.init_results_dict()

    model.eval()
    with torch.no_grad():
        end = time.time()
        runner.before_inference(model, tokenizer, device, result_dict)
        for index, payload in enumerate(tqdm(valid_loader)):
            # compute output
            runner.inference(model, payload, tokenizer, device, result_dict)

            # measure batch time
            batch_time.update(time.time() - end)
            end = time.time()

            if args.debug:
                break

    if args.write_results:
        with open(f'{args.checkpoint_path}/result-valid.json', 'w') as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=4)

    if args.debug:
        logger.info(json.dumps(result_dict, indent=2, ensure_ascii=False))
    try:
        runner.measure_result(result_dict, meters)
    except Exception as e:
        logger.info(f'Measure Result Error: {repr(e)}')
    progress.display(0)
    if args.mode == 'train' and not args.debug:
        writer.add_scalars('meter valid', {field_name: meter.avg for field_name, meter in meters.items()}, epoch)
    return meters


def test(test_loader, tokenizer, model, device, args):
    batch_time = AverageMeter('Time', ':6.3f')
    meters = runner.init_meters_dict()
    progress = ProgressMeter(
        logger,
        len(test_loader),
        [batch_time] + list(meters.values()),
        prefix='Test: '
    )

    result_dict = runner.init_results_dict()

    model.eval()
    with torch.no_grad():
        end = time.time()
        runner.before_inference(model, tokenizer, device, result_dict)
        for index, payload in enumerate(tqdm(test_loader)):
            # compute output
            runner.inference(model, payload, tokenizer, device, result_dict)

            # measure batch time
            batch_time.update(time.time() - end)
            end = time.time()

            if args.debug:
                break

    if args.write_results:
        with open(f'{args.checkpoint_path}/result-test.json', 'w') as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=4)

    if args.debug:
        logger.info(json.dumps(result_dict, indent=2, ensure_ascii=False))

    if args.measure_test:
        try:
            runner.measure_result(result_dict, meters)
        except Exception as e:
            logger.info(f'Measure Result Error: {repr(e)}')
        progress.display(0)


if __name__ == '__main__':
    options = Options()
    args = options.parse()
    args.n_gpu = torch.cuda.device_count()
    args.pretrained_path = f'{args.home}/{args.code}/pretrained/'

    logger = Logger(logging.getLogger(__name__), args.logging)

    # set up random seed
    torch.manual_seed(args.seed)

    if args.n_gpu > 1:
        torch.cuda.set_device(args.local_rank)
        init_process_group(
            backend="nccl",
            init_method='env://',
            world_size=args.n_gpu,
            rank=args.local_rank,
            timeout=datetime.timedelta(hours=1.5)
        )

    with torch_distributed_main_first(args.local_rank):

        utils.mk_dir(os.path.join(args.home, args.code, args.checkpoint_dir, args.name))
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s: %(message)s',
            filename=os.path.join(args.home, args.code, args.checkpoint_dir, args.name, 'log'),
            filemode='a',
            level=logging.INFO
        )

        if args.code == 'transformer':
            runner = TransformerRunner(args)
        elif args.code == 'graph':
            # runner = GraphRunner(args)
            pass
        elif args.code == 'bart':
            runner = BartRunner()
        elif args.code == 'unigdd':
            runner = UniGddRunner(args)
        elif args.code == 'fid':
            runner = FiDRunner(args)
        elif args.code == 'universe':
            runner = UniverseUniGddRunner(args)
        elif args.code == 'longformer':
            runner = LongFormerRunner(args)
        elif args.code == 'treejc':
            runner = TreeJCRunner(args)
        elif args.code == 'dpr':
            runner = DPRRunner(args)
        elif args.code == 'rag':
            runner = RAGRunner(args)
        elif args.code == 'struc-fid':
            runner = StrucFiDRunner(args)
        elif args.code == 'mt5':
            runner = MT5Runner(args)
        elif args.code == 'dpl':
            runner = PolicyRunner(args)
        elif args.code == 'act':
            runner = ActRunner(args)
        elif args.code == 'sfre2g':
            runner = SFRe2GRunner(args)
        else:
            raise Exception(f'Undefined run code: {args.code}')

    main_work(args, runner)
