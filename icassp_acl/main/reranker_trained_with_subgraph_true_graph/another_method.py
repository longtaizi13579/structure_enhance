import os
import sys

sys.path.append(os.getcwd())
import logging
import random
import transformers
import ujson as json
from util.line_corpus import jsonl_lines, block_shuffle
from another_model_for_ddqn import RerankerHypers
from ddqn_model import DQNAgent

transformers.logging.set_verbosity_error()
logger = logging.getLogger(__name__)


class RerankerTrainArgs(RerankerHypers):
    def __init__(self):
        super().__init__()
        # MARK
        self.instances_size = 1
        self.positive_pids = ''

        self.positive_pids = 'data_dir/multidoc2dial_section/multidoc2dial_train_positive_pids.jsonl'
        self.output_dir = 'models/output_test'
        self.initial_retrieval = 'data_dir/multidoc2dial_section/train_merge.jsonl'
        self.max_steps = 8
        self.batch_size = 2  # 32


class RerankerTrainer:
    def __init__(self, args: RerankerTrainArgs):
        fold_num, fold_count = args.kofn(args.fold)
        # load id to positive pid map
        self.inst_id2pos_pids = dict()
        self.inst_id2pos_passages = dict()
        for line in jsonl_lines(args.positive_pids):
            jobj = json.loads(line)
            self.inst_id2pos_pids[jobj['id']] = jobj['positive_pids']
            if args.add_all_positives:
                self.inst_id2pos_passages[jobj['id']] = jobj['positive_passages']
            assert isinstance(jobj['positive_pids'], list)
        logger.info(f'gathered positive pids for {len(self.inst_id2pos_pids)} instances')

        # remove out-of-recall
        instance_count = 0
        for line in jsonl_lines(args.initial_retrieval):
            jobj = json.loads(line)
            inst_id = jobj['id']
            if inst_id not in self.inst_id2pos_pids:
                continue
            passages = jobj['passages']
            positive_pids = self.inst_id2pos_pids[inst_id]
            target_mask = [p['pid'] in positive_pids for p in passages]
            if (not args.add_all_positives and not any(target_mask)) or all(target_mask):
                del self.inst_id2pos_pids[inst_id]
            else:
                instance_count += 1
        if instance_count != len(self.inst_id2pos_pids):
            logger.error(f'!!! Mismatch between --positive_pids and --initial_retrieval! '
                         f'{len(self.inst_id2pos_pids)} vs {instance_count}')
        self.args = args
        self.max_length_count = 0

    def passage_correctness(self, pid, positive_pids, positive_dids):
        if pid in positive_pids:
            return 1.0
        elif positive_dids and pid[:pid.index('::')] in positive_dids:
            return self.args.doc_match_weight
        else:
            return 0

    def train(self):
        rand = random.Random()
        agent = DQNAgent()
        for every_epoch in range(self.args.num_train_epochs):
            dataset = block_shuffle(jsonl_lines(self.args.initial_retrieval), block_size=100000, rand=rand)
            for line_ndx, line in enumerate(dataset):
                jobj = json.loads(line)
                inst_id = jobj['id']
                if inst_id not in self.inst_id2pos_pids:
                    continue
                if line_ndx % self.args.world_size != self.args.global_rank:
                    continue
                query = jobj['input'] if 'input' in jobj else jobj['query']
                passages = jobj['passages']
                positive_pids = self.inst_id2pos_pids[inst_id]
                agent.reset(passages, positive_pids)
                state = query
                for step in range(self.args.max_steps):
                    action, actions = agent.get_action(state)
                    reward, done, next_actions = agent.step(action)
                    agent.replay_buffer.push(state, action, actions, next_actions, reward, done)
                    if len(agent.replay_buffer) > self.args.batch_size:
                        agent.update(self.args.batch_size)
                    if done:
                        break
        logger.info(f'loss_history = {self.loss_history.loss_history}')
        logger.info(f'truncated to max length ({self.args.max_seq_length}) {self.max_length_count} times')
        agent.save_transformer()


def main():
    args = RerankerTrainArgs()
    args.fill_from_args()
    args.set_seed()
    assert args.full_train_batch_size % args.world_size == 0
    args.gradient_accumulation_steps = args.full_train_batch_size // (args.per_gpu_train_batch_size * args.world_size)
    trainer = RerankerTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
