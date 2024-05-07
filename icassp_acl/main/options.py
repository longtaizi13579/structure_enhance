import argparse
import logging
from pathlib import Path


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialize_parser()
        self.add_train_options()
        self.add_eval_options()
        self.add_debug_options()
        self.add_generation_options()
        self.add_customer_options()

    def initialize_parser(self):
        # code parameters
        self.parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
        self.parser.add_argument('--home', type=str, default='/mnt/data/huairang/Doc2dial/', help='home path')
        self.parser.add_argument('--code', type=str, default='transformer', help='code to run')
        self.parser.add_argument('--mode', type=str, default='train', help='train, valid or test')

        # pre-trained model parameters
        self.parser.add_argument('--model_path', type=str, default='none', help='path for retraining')
        self.parser.add_argument('--model', type=str, default=None, help='pre-trained model')

        # checkpoint parameters
        self.parser.add_argument('--checkpoint_dir', type=str, default='checkpoint', help='models are saved here')
        self.parser.add_argument('--name', type=str, default='transformer', help='name of the experiment')
        self.parser.add_argument('--init_state_dict_from', type=str, default='',
                                 help='name of state dict initialization checkpoint')

        # dataset parameters
        self.parser.add_argument('--reload', action='store_true', help="reload cached dataset")
        self.parser.add_argument("--dataset", default="data", type=str, help="use cached dataset")
        self.parser.add_argument("--dataset_prefix", default="", type=str, help="prefix of dataset")
        self.parser.add_argument('--reload_train_per_epoch', action='store_true',
                                 help="reload train dataset per epoch")

        # gpu parameters
        self.parser.add_argument("--local_rank", type=int, default=-1,
                                 help="For distributed training: local_rank")
        self.parser.add_argument("--no_find_unused_parameters", action='store_false',
                                 help='do not find unused parameters in model')
        self.parser.add_argument("--main_port", type=int, default=-1,
                                 help="Main port (for multi-node SLURM jobs)")

        # logger parameters
        self.parser.add_argument('--logging', type=str, default='both')

    def add_train_options(self):
        # epoch
        self.parser.add_argument("--start_epoch", default=0, type=int, help="Start form which epoch.")
        self.parser.add_argument('--epochs', type=int, default=10, help="total epochs")

        # basic
        self.parser.add_argument('--lr', type=float, default=2e-5, help='learning rate')
        self.parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
        self.parser.add_argument("--per_gpu_batch_size", default=16, type=int,
                                 help="Batch size per GPU/CPU for training.")
        self.parser.add_argument('--accumulation_steps', type=int, default=1)
        self.parser.add_argument('--gradient_checkpoint_segments', type=int, default=None)

        # optimizer
        self.parser.add_argument("--eps", default=1e-8, type=float, help="Epsilon for AdamW optimizer")
        self.parser.add_argument('--max_grad_norm', type=float, default=None)
        self.parser.add_argument('--weight_decay', type=float, default=0.1)

        # scheduler
        self.parser.add_argument('--scheduler', type=str, default='linear', help='scheduler type')
        self.parser.add_argument('--warmup_steps', type=int, default=1000)
        self.parser.add_argument('--warmup_rate', type=float, default=None)
        self.parser.add_argument('--print_freq', type=int, default=40)

    def add_eval_options(self):
        self.parser.add_argument('--eval_batch_size', type=int, default=32,
                                 help='Batch size per GPU/CPU for evaluation')
        self.parser.add_argument('--skip_eval_dur_train', action='store_true',
                                 help='skip eval performance during train')
        self.parser.add_argument('--start_eval_epoch', type=int, default=0,
                                 help='start eval performance epoch during train')
        self.parser.add_argument('--measure_test', default=True, help='measure test results')
        self.parser.add_argument('--write_results', action='store_true', default=True, help='save results')

    def add_generation_options(self):
        self.parser.add_argument('--source_sequence_size', type=int, default=2560,
                                 help='Input sequence size of encoder')
        self.parser.add_argument('--target_sequence_size', type=int, default=500,
                                 help='Output sequence size of decoder')
        self.parser.add_argument('--beam_size', type=int, default=3,
                                 help='Output sequence size of decoder')

    def add_debug_options(self):
        self.parser.add_argument('--debug', action='store_true', help='Open debug mode')
        self.parser.add_argument('--unit_test', action='store_true', help='Open unit test mode')

    def add_customer_options(self):
        self.parser.add_argument('--copy', action='store_true', help='copy mode')
        self.parser.add_argument('--no_pretrain', action='store_true', help='no pretrain model')
        self.parser.add_argument('--add_parent', action='store_true', help='concatenate parent node')
        self.parser.add_argument('--data_type', type=str, default='all', help='dataset type')
        self.parser.add_argument('--language', type=str, default='en', help='language')
        self.parser.add_argument('--passages', type=int, default=5, help='passage number for rerank')
        self.parser.add_argument('--num_labels', type=int, default=None, help='num labels')
        self.parser.add_argument('--rerank_path', type=str, default='', help='rerank model path')
        self.parser.add_argument('--subgraph', action='store_true', help='add sub-graph')
        self.parser.add_argument('--rm_struct', action='store_true', help='remove_structure')

    def parse(self):
        opt = self.parser.parse_args()
        return opt
