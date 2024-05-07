import os
import sys
#sys.path.append(os.getcwd())
os.chdir('/mnt/workspace/yeqin/huihuan/acl_code')
sys.path.append('/mnt/workspace/yeqin/huihuan/acl_code')
import logging
from torch_util.transformer_optimize import LossHistory, TransformerOptimize
from reranker_trained_with_subgraph_true_graph.another_model import load, save_transformer, RerankerHypers
import ujson as json
from util.line_corpus import jsonl_lines, block_shuffle, write_open
import torch.nn.functional as F
import random
import os
import torch
import transformers
import copy
from torch.utils.checkpoint import checkpoint
from torch import nn
#from torch_geometric.nn import RGATConv, RGCNConv
from differ_GAT_network import myGATConv
transformers.logging.set_verbosity_error()
logger = logging.getLogger(__name__)

class EncoderWrapper(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask, dummy_tensor):
        whole_output = self.encoder(input_ids, attention_mask).pooler_output
        return whole_output


class RerankerTrainArgs(RerankerHypers):
    def __init__(self):
        super().__init__()
        # MARK
        self.instances_size = 1
        self.positive_pids = ''
        #self.__required_args__ = ['positive_pids', 'output_dir', 'initial_retrieval']

        self.positive_pids = 'data_dir/multidoc2dial_section/multidoc2dial_train_positive_pids.jsonl'
        self.output_dir = 'models/output_test'
        self.initial_retrieval = 'data_dir/multidoc2dial_section/train_merge.jsonl'

class Two_level_Model(nn.Module):
    def __init__(self, model):
        super(Two_level_Model, self).__init__()
        #self.dropout = model.dropout
        self.passage_classifier = nn.Linear(2048, 2)
        self.graph_classifier = nn.Linear(2048, 2)
        self.bert = EncoderWrapper(model)
        #self.subgraph_classifier = copy.deepcopy(model.classifier)
        self.device = model.device
        self.encoder_gpu_train_limit = 10
        self.conv = myGATConv(1024, 1024, add_self_loops=False)#(1024, 1024, 2)


    def encode(self, model, passages):
        input_ids = passages['input_ids']
        attention_mask = passages['attention_mask']
        dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)
        all_pooled_output = []
        for sub_bndx in range(0, input_ids.shape[0], self.encoder_gpu_train_limit):
            sub_input_ids = input_ids[sub_bndx:sub_bndx + self.encoder_gpu_train_limit]
            sub_attention_mask = attention_mask[sub_bndx:sub_bndx + self.encoder_gpu_train_limit]
            pooler_output = checkpoint(model, sub_input_ids, sub_attention_mask, dummy_tensor)
            all_pooled_output.append(pooler_output)
        return torch.cat(all_pooled_output, dim=0)

    
    def forward(self, passages, subgraph_id_group, passage_id_group, now_edge, now_edge_type):
        get_passage_length = len(passages['input_ids'])
        whole_outputs = self.encode(self.bert, passages)
        #all_pooled_output = whole_outputs
        graph_enhance = self.conv(whole_outputs, now_edge)#, now_edge_type)
        passage_enhance = graph_enhance[passage_id_group]
        subgraph_enhance = graph_enhance[subgraph_id_group]
        passage_enhance = torch.cat([passage_enhance, whole_outputs[passage_id_group]], dim=-1)
        subgraph_enhance = torch.cat([subgraph_enhance, whole_outputs[subgraph_id_group]], dim=-1)
        #passage_pooled_output = self.dropout(passage_pooled_output)
        passage_logits = self.passage_classifier(passage_enhance)
        subgraph_logits = self.graph_classifier(subgraph_enhance)
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
        now_subgraphs_id = copy.deepcopy(subgraphs_id)
        for every_cluster in now_subgraphs_id:
            every_cluster.insert(0, '-1')
        query_ids = self.tokenizer([query], add_special_tokens=False, return_tensors='pt')['input_ids'][0][:195]
        query = self.tokenizer.decode(query_ids)
        # concate
        texts_b = []
        texts_c = []
        new_subgraphs = []
        now_edge = []
        now_edge_type = []
        now_ptr = 0
        whole_relative_id = []
        all_distinction = []
        for every_cluster in subgraphs:
            now_cluster = []
            max_common_words = ''
            for every_passage in every_cluster:
                every_passage = every_passage.replace('[SEP]', '\\')
                now_cluster.append(every_passage)
                if len(max_common_words):
                    now_slide_one = ''
                    now_slide_two = ''
                    ptr = 1
                    while now_slide_one == now_slide_two and len(now_slide_one) != len(max_common_words):
                        ptr += 1
                        now_slide_one = max_common_words[:ptr]
                        now_slide_two = every_passage[:ptr]
                    max_common_words = max_common_words[:ptr-1] + 'x'
                else:
                    max_common_words = every_passage + 'x'
            max_common_words = '<' + max_common_words[:-1].strip(' <3> ').strip(' <2> ')
            new_cluster = [max_common_words]
            new_cluster.extend(now_cluster)
            distinction = list(range(now_ptr, now_ptr+len(new_cluster)))
            all_distinction.append(now_ptr+len(new_cluster))
            # 双向聚合
            first_one_flag = 1
            for every_one_dist in distinction:
                if first_one_flag == 1:
                    for every_one_other in distinction:
                        # 去除自环
                        if every_one_dist != every_one_other:
                            now_edge.append([every_one_dist, every_one_other])
                            # summarization 
                            now_edge_type.append(1)
                    first_one_flag = 0
                else:
                    for every_one_other in distinction:
                        # 去除自环
                        if every_one_dist != every_one_other:
                            now_edge.append([every_one_dist, every_one_other])
                            now_edge_type.append(0)
            whole_relative_id.append(distinction)
            #now_edge.append(distinction)
            now_ptr = now_ptr+len(new_cluster)
            new_subgraphs.append(new_cluster)
        subgraphs = new_subgraphs
        expand_to_passage = []
        for every_cluster in subgraphs:
            for every_passage in every_cluster:
                expand_to_passage.append(every_passage)
        print()
        # print()
        
        # for every_cluster_index in range(len(subgraphs)):
        #     now_cluster = subgraphs[every_cluster_index]
        #     now_sentence = ''
        #     for every_passage_index in range(len(now_cluster)):
        #         core_passage = now_cluster[every_passage_index]
        #         other_passages = []
        #         # order = list(range(len(now_cluster)))
        #         # random.shuffle(order)
        #         #random select
        #         # random_num = random.randint(1, len(order))
        #         # order = order[:random_num]
        #         # shuffle 
        #         for every_passage_internal_index in range(len(now_cluster)):
        #             if  every_passage_internal_index != every_passage_index:
        #                 other_passages.append(now_cluster[every_passage_internal_index])
        #         others = ' [SEP] '.join(other_passages)
        #         texts_b.append(core_passage + ' [others] ' + others)
        #     texts_c.append(' [SEP] '.join(now_cluster))
        texts_a = [query] * len(expand_to_passage)
        # MARK
        #texts_b = [p['text'] for p in passages]
        inputs = self.tokenizer(
            texts_a, expand_to_passage,
            add_special_tokens=True,
            return_tensors='pt',
            max_length=self.args.max_seq_length,
            padding='longest',
            truncation=True)

        # texts_a = [query] * len(texts_c)
        # inputs_subgraph = self.tokenizer(
        #     texts_a, texts_c,
        #     add_special_tokens=True,
        #     return_tensors='pt',
        #     max_length=self.args.max_seq_length,
        #     padding='longest',
        #     truncation=True)
        # # track how often we truncate to max_seq_length
        # if inputs['input_ids'].shape[1] == self.args.max_seq_length:
        #     self.max_length_count += 1
        inputs = {n: t.to(model.device) for n, t in inputs.items()}
        # inputs_subgraph = {n: t.to(model.device) for n, t in inputs_subgraph.items()}
        ptr_mem = 0
        subgraph_id_group = []
        passage_id_group = []
        for every_cluster_id in now_subgraphs_id:
            for every_internal_id in every_cluster_id:
                if every_internal_id == '-1':
                    subgraph_id_group.append(ptr_mem)
                else:
                    passage_id_group.append(ptr_mem)
                ptr_mem += 1
        passage_logits, subgraph_logits = model(inputs, torch.tensor(subgraph_id_group).cuda(), torch.tensor(passage_id_group).cuda(), torch.tensor(now_edge).transpose(0, 1).cuda(), torch.tensor(now_edge_type).cuda())
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
                nll = -(passage_logprobs.dot(torch.tensor(correctness).to(passage_logprobs.device)))
                #subgraph_nll = -(subgraph_logprobs.dot(torch.tensor(subgraph_correctness).to(subgraph_logprobs.device)))
                #nll = passage_nll + subgraph_nll
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
