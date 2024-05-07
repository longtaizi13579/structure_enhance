import os
import sys
sys.path.append(os.getcwd())
# os.chdir('/mnt/workspace/yeqin/huihuan/acl_code')
# sys.path.append('/mnt/workspace/yeqin/huihuan/acl_code')
import logging
from torch_util.transformer_optimize import LossHistory, TransformerOptimize
from reranker_trained_with_subgraph.another_model import load, save_transformer, RerankerHypers
import ujson as json
from util.line_corpus import jsonl_lines, block_shuffle, write_open
import torch.nn.functional as F
import random
import os
import torch
import transformers
import copy
from torch import nn
transformers.logging.set_verbosity_error()
logger = logging.getLogger(__name__)


class RerankerTrainArgs(RerankerHypers):
    def __init__(self):
        super().__init__()
        # MARK
        self.instances_size = 1
        self.positive_pids = ''
        self.__required_args__ = ['positive_pids', 'output_dir', 'initial_retrieval']

        # self.positive_pids = 'doc2bot_train_positive_pids.jsonl'
        # self.output_dir = 'models/rerank_test'
        # self.initial_retrieval = 'predictions/dprKGI0/merge.jsonl'

class Two_level_Model(nn.Module):
    def __init__(self, model):
        super(Two_level_Model, self).__init__()
       #self.dropout = model.dropout
        self.classifier = model.classifier
        self.bert = model.roberta
        self.subgraph_classifier = copy.deepcopy(model.classifier)
        self.device = model.device


    def forward(self, passages, subgraphs):
        passage_outputs = self.bert(**passages)
        subgraph_outputs = self.bert(**subgraphs)
        passage_pooled_output = passage_outputs[0]
        #passage_pooled_output = self.dropout(passage_pooled_output)
        passage_logits = self.classifier(passage_pooled_output)
        subgraph_pooled_output = subgraph_outputs[0]
        #subgraph_pooled_output = self.dropout(subgraph_pooled_output)
        subgraph_logits = self.subgraph_classifier(subgraph_pooled_output)
        return F.log_softmax(passage_logits, dim=-1)[:, 1], F.log_softmax(subgraph_logits, dim=-1)[:, 1]

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
        if fold_count > 1:
            inst_ids = list(self.inst_id2pos_pids.keys())
            inst_ids.sort()
            fold_inst_ids = set(inst_ids[fold_num::fold_count])
            self.inst_id2pos_pids = \
                {inst_id: pos_pids for inst_id, pos_pids in self.inst_id2pos_pids.items() if inst_id not in fold_inst_ids}
            instance_count = len(self.inst_id2pos_pids)
            with write_open(os.path.join(args.output_dir, 'trained_on_instances.json')) as f:
                json.dump(list(self.inst_id2pos_pids.keys()), f)
        assert instance_count == len(self.inst_id2pos_pids)

        # load model and tokenizer
        origin_model, self.tokenizer = load(args)
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["[unused1]", "[unused2]", "[unused3]", "[unused4]"]})
        model = Two_level_Model(origin_model)
        # whole model
        model.to(model.device)
        # transformer_optimize
        if args.train_instances <= 0:
            args.train_instances = instance_count
        # MARK
        instances_to_train_over = args.train_instances * args.num_train_epochs // args.instances_size
        self.optimizer = TransformerOptimize(args, instances_to_train_over, model)
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        self.optimizer.model.zero_grad()
        # MARK
        self.loss_history = LossHistory(args.train_instances //
                                   (args.full_train_batch_size // args.gradient_accumulation_steps) // args.instances_size)
        self.args = args
        self.max_length_count = 0
    
    def one_batch(self, query_batch, passages_batch):
        model = self.optimizer.model
        texts_a = []
        texts_b = []
        for query, passages in zip(query_batch, passages_batch):
            texts_a += [query] * len(passages)
            texts_b += [p['title'] + '\n\n' + p['text'] for p in passages]
        #     pass
        # texts_a = [query] * len(passages)
        # texts_b = [p['title'] + '\n\n' + p['text'] for p in passages]
        inputs = self.tokenizer(
            texts_a, texts_b,
            add_special_tokens=True,
            return_tensors='pt',
            max_length=self.args.max_seq_length,
            padding='longest',
            truncation=True)
        # track how often we truncate to max_seq_length
        if inputs['input_ids'].shape[1] == self.args.max_seq_length:
            self.max_length_count += 1
        inputs = {n: t.to(model.device) for n, t in inputs.items()}
        logits = F.log_softmax(model(**inputs)[0], dim=-1)[:, 1]  # log_softmax over the binary classification
        logprobs = F.log_softmax(logits, dim=0)  # log_softmax over the passages
        # we want the logits rather than the logprobs as the teacher labels
        return logprobs
    
    def one_instance(self, query, passages, subgraphs, subgraphs_id):
        model = self.optimizer.model
        # special_token_id = self.tokenizer.added_tokens_encoder['[special]']
        query_ids = self.tokenizer([query], add_special_tokens=False, return_tensors='pt')['input_ids'][0][:195]
        query = self.tokenizer.decode(query_ids)

        # concate
        texts_b = []
        texts_c = []
        new_subgraphs = []
        for every_cluster in subgraphs:
            now_cluster = []
            for every_passage in every_cluster:
                every_passage = every_passage.replace('[SEP]', '[unused1]')
                now_cluster.append(every_passage)
            new_subgraphs.append(now_cluster)
        subgraphs = new_subgraphs
        for every_cluster_index in range(len(subgraphs)):
            now_cluster = subgraphs[every_cluster_index]
            now_sentence = ''
            for every_passage_index in range(len(now_cluster)):
                core_passage = now_cluster[every_passage_index]
                other_passages = []
                # order = list(range(len(now_cluster)))
                # random.shuffle(order)
                #random select
                # random_num = random.randint(1, len(order))
                # order = order[:random_num]
                # shuffle 
                for every_passage_internal_index in range(len(now_cluster)):
                    if  every_passage_internal_index != every_passage_index:
                        other_passages.append(now_cluster[every_passage_internal_index])
                        
                other_passages = [other_passage.split('[unused1]')[0] for other_passage in other_passages]
                others = '[unused2]'.join(other_passages)
                texts_b.append('[unused3]' +core_passage + '[unused4]' + others)

            texts_c.append('[unused1]'.join(now_cluster))
        texts_a = [query] * len(texts_b)
        # MARK
        #texts_b = [p['text'] for p in passages]
        inputs = self.tokenizer(
            texts_a, texts_b,
            add_special_tokens=True,
            return_tensors='pt',
            max_length=self.args.max_seq_length,
            padding='longest',
            truncation=True)

        texts_a = [query] * len(texts_c)
        inputs_subgraph = self.tokenizer(
            texts_a, texts_c,
            add_special_tokens=True,
            return_tensors='pt',
            max_length=self.args.max_seq_length,
            padding='longest',
            truncation=True)
        # track how often we truncate to max_seq_length
        if inputs['input_ids'].shape[1] == self.args.max_seq_length:
            self.max_length_count += 1
        inputs = {n: t.to(model.device) for n, t in inputs.items()}
        inputs_subgraph = {n: t.to(model.device) for n, t in inputs_subgraph.items()}

        passage_logits, subgraph_logits = model(inputs, inputs_subgraph)
        passage_logprobs = F.log_softmax(passage_logits, dim=0)  # log_softmax over the passages
        subgraph_logprobs = F.log_softmax(subgraph_logits, dim=0)  # log_softmax over the passages
        # we want the logits rather than the logprobs as the teacher labels
        return passage_logprobs, subgraph_logprobs

    def limit_gpu_sequences_binary(self, passages, target_mask, rand):
        if len(passages) > self.args.max_num_seq_pairs_per_device:
            num_pos = min(sum(target_mask), self.args.max_num_seq_pairs_per_device // 2)
            num_neg = self.args.max_num_seq_pairs_per_device - num_pos
            passage_and_pos = list(zip(passages, target_mask))
            rand.shuffle(passage_and_pos)
            pos_count = 0
            neg_count = 0
            passages = []
            target_mask = []
            for passage, mask in passage_and_pos:
                if mask and pos_count < num_pos:
                    passages.append(passage)
                    target_mask.append(mask)
                    pos_count += 1
                elif not mask and neg_count < num_neg:
                    passages.append(passage)
                    target_mask.append(mask)
                    neg_count += 1
        return passages, target_mask

    def limit_gpu_sequences(self, passages, correctness, rand):
        if len(passages) > self.args.max_num_seq_pairs_per_device:
            num_pos = min(sum([c > 0 for c in correctness]), self.args.max_num_seq_pairs_per_device // 2)
            num_neg = self.args.max_num_seq_pairs_per_device - num_pos
            passage_and_pos = list(zip(passages, correctness))
            rand.shuffle(passage_and_pos)
            pos_count = 0
            neg_count = 0
            passages = []
            correctness = []
            for passage, pos in passage_and_pos:
                if pos > 0 and pos_count < num_pos:
                    passages.append(passage)
                    correctness.append(pos)
                    pos_count += 1
                elif pos == 0 and neg_count < num_neg:
                    passages.append(passage)
                    correctness.append(pos)
                    neg_count += 1
        return passages, correctness

    def passage_correctness(self, pid, positive_pids, positive_dids):
        if pid in positive_pids:
            return 1.0
        elif positive_dids and pid[:pid.index('::')] in positive_dids:
            return self.args.doc_match_weight
        else:
            return 0

    def train(self):
        rand = random.Random()
        while self.optimizer.should_continue():
            self.optimizer.model.train()
            dataset = block_shuffle(jsonl_lines(self.args.initial_retrieval), block_size=100000, rand=rand)
            # query_batch = []
            # passages_batch = []
            # correctness_batch = []
            for line_ndx, line in enumerate(dataset):
                jobj = json.loads(line)
                inst_id = jobj['id']
                if inst_id not in self.inst_id2pos_pids:
                    continue
                if line_ndx % self.args.world_size != self.args.global_rank:
                    continue
                query = jobj['input'] if 'input' in jobj else jobj['query']
                passages = jobj['passages']
                
                
                subgraphs = jobj['subgraphs']
                subgraphs_id = jobj['subgraphs_id']

                
                if self.args.add_all_positives:
                    add_pos_passages = self.inst_id2pos_passages[inst_id]
                    passages.extend([p for p in add_pos_passages if p['pid'] not in passages])
                positive_pids = self.inst_id2pos_pids[inst_id]
                if self.args.doc_match_weight > 0:
                    positive_dids = [pid[:pid.index('::')] for pid in positive_pids]
                else:
                    positive_dids = None
                # target_mask = [p['pid'] in positive_pids for p in passages]
                # passages, target_mask = self.limit_gpu_sequences(passages, target_mask, rand)
                correctness = [self.passage_correctness(p['pid'], positive_pids, positive_dids) for p in passages]
                passages, correctness = self.limit_gpu_sequences(passages, correctness, rand)
                correctness = []
                subgraph_correctness = []
                for pid_cluster in subgraphs_id:
                    flag = 0.0
                    for every_pid in pid_cluster:
                        if every_pid in positive_pids:
                            correctness.append(1.0)
                            flag = 1.0
                        else:
                            correctness.append(0.0)
                    subgraph_correctness.append(flag)
                passage_logprobs, subgraph_logprobs = self.one_instance(query, passages, subgraphs, subgraphs_id)
                passage_nll = -(passage_logprobs.dot(torch.tensor(correctness).to(passage_logprobs.device)))
                subgraph_nll = -(subgraph_logprobs.dot(torch.tensor(subgraph_correctness).to(subgraph_logprobs.device)))
                nll = passage_nll + subgraph_nll
                loss_val = self.optimizer.step_loss(nll)
                get_best = self.loss_history.note_loss(loss_val)
                if get_best == 2:
                    save_transformer(self.args, self.optimizer.model, self.tokenizer, save_dir=self.args.output_dir+'_best')
                if not self.optimizer.should_continue():
                    break

        logger.info(f'loss_history = {self.loss_history.loss_history}')
        logger.info(f'truncated to max length ({self.args.max_seq_length}) {self.max_length_count} times')
        save_transformer(self.args, self.optimizer.model, self.tokenizer)


def main():
    args = RerankerTrainArgs()
    args.fill_from_args()
    args.set_seed()
    assert args.full_train_batch_size % args.world_size == 0
    # assert args.n_gpu == 1
    args.gradient_accumulation_steps = args.full_train_batch_size // (args.per_gpu_train_batch_size * args.world_size)

    trainer = RerankerTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()